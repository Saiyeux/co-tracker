# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np
import ffmpeg
import pynvml
from PIL import Image
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor
import time
import sys
import matplotlib.pyplot as plt  # 新增：用于绘制查询点

log_file = open("output.log", "a")
sys.stdout = log_file

# 记录整个过程的开始时间
start_time = time.time()

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# 初始化 NVML 并获取 GPU 占用量的函数
def initialize_nvml():
    try:
        pynvml.nvmlInit()
        if torch.cuda.is_available():
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return handle
        print("No CUDA GPU available or NVML initialization failed, skipping GPU monitoring")
        return None
    except pynvml.NVMLError as e:
        print(f"NVML initialization failed: {e}, skipping GPU monitoring")
        return None

def get_gpu_memory(handle):
    if handle is None:
        return None, None
    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = mem_info.used / 1024 / 1024
        total_mb = mem_info.total / 1024 / 1024
        return used_mb, total_mb
    except pynvml.NVMLError as e:
        print(f"Failed to get GPU memory info: {e}")
        return None, None

# 分块读取视频的函数
def read_video_with_ffmpeg_chunked(video_path, chunk_size=100):
    print(f"Starting to read video: {video_path}")
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        total_frames = int(video_info['nb_frames'])
        print(f"Video info: {width}x{height}, total frames: {total_frames}")
        
        process = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )
        
        frames = []
        frame_count = 0
        while True:
            in_bytes = process.stdout.read(width * height * 3)
            if not in_bytes:
                if frames:
                    chunk = np.stack(frames, axis=0)
                    print(f"Yielding final chunk with {chunk.shape[0]} frames")
                    yield chunk
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))
            frames.append(frame)
            frame_count += 1
            
            if len(frames) >= chunk_size:
                chunk = np.stack(frames, axis=0)
                print(f"Yielding chunk with {chunk.shape[0]} frames at frame {frame_count - chunk_size}")
                yield chunk
                frames = []
        
        process.stdout.close()
        process.wait()
        print("Video reading completed")
    
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise
    except Exception as e:
        print(f"Error reading video: {e}")
        raise

# 新增：绘制查询点位置的函数
def plot_query_points(queries, video_shape, save_dir, chunk_idx):
    frame_numbers = queries[:, 0].int().tolist()
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    
    for i, (query, frame_number) in enumerate(zip(queries, frame_numbers)):
        ax = axs[i]
        ax.plot(query[1].item(), query[2].item(), 'ro')
        ax.set_title(f"Frame {frame_number}")
        ax.set_xlim(0, video_shape[4])
        ax.set_ylim(0, video_shape[3])
        ax.invert_yaxis()
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"query_points_chunk_{chunk_idx}.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Query points plot saved to {plot_path}")
    return plot_path

# 新增：记录跟踪点位置的函数
def log_track_positions(pred_tracks, chunk_idx, global_frame_offset):
    print(f"\nLogging track positions for chunk {chunk_idx}")
    for point_idx in range(pred_tracks.shape[2]):  # 遍历每个跟踪点
        for frame_idx in range(pred_tracks.shape[1]):  # 遍历每个帧
            x = pred_tracks[0, frame_idx, point_idx, 0].item()
            y = pred_tracks[0, frame_idx, point_idx, 1].item()
            global_frame = global_frame_offset + frame_idx
            print(f"Chunk {chunk_idx}, Frame {global_frame}, Point {point_idx}: (x={x:.2f}, y={y:.2f})")

if __name__ == "__main__":
    print("\n" + '-' * 10 + " Section starts here " + '-' * 10)
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="/home/surgicalai/Data/test5.mp4", help="path to a video")
    parser.add_argument("--mask_path", default="./assets/apple_mask.png", help="path to a segmentation mask")
    parser.add_argument("--checkpoint", default=None, help="CoTracker model parameters")
    parser.add_argument("--grid_size", type=int, default=20, help="Regular grid size")
    parser.add_argument("--grid_query_frame", type=int, default=0, help="Compute dense and grid tracks starting from this frame")
    parser.add_argument("--backward_tracking", action="store_true", help="Compute tracks in both directions")
    parser.add_argument("--use_v2_model", action="store_true", help="Use CoTracker2 instead of CoTracker++")
    parser.add_argument("--offline", action="store_true", help="Use the offline model")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of frames per chunk")
    parser.add_argument("--save_dir", default="/home/surgicalai/Data/output/co-tracker", help="Directory to save output videos")

    args = parser.parse_args()
    print(f"Arguments parsed: video_path={args.video_path}, chunk_size={args.chunk_size}, save_dir={args.save_dir}")

    # 新增：定义查询点
    queries = torch.tensor([
        [0., 850., 450.],  # point tracked from the first frame
        [10., 900., 600.], # frame number 10
        [20., 950., 700.], # ...
        [30., 1050., 500.]
    ])
    if torch.cuda.is_available():
        queries = queries.cuda()
    print(f"Queries defined, shape: {queries.shape}, device: {queries.device}")

    # 初始化 NVML
    gpu_handle = initialize_nvml()

    # 加载分割掩码（注释掉未使用的部分）
    # print(f"Loading segmentation mask from {args.mask_path}")
    # segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
    # segm_mask = torch.from_numpy(segm_mask)[None, None]
    # print(f"Mask loaded, shape: {segm_mask.shape}")

    # 初始化模型
    print("Initializing CoTracker model")
    if args.checkpoint is not None:
        if args.use_v2_model:
            model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
        else:
            window_len = 60 if args.offline else 16
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=args.use_v2_model,
                offline=args.offline,
                window_len=window_len,
            )
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(DEFAULT_DEVICE)
    print(f"Model initialized and moved to {DEFAULT_DEVICE}")

    # 分块处理视频
    seq_name = args.video_path.split("/")[-1].split(".")[0]
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory prepared: {save_dir}")
    
    chunk_idx = 0
    all_tracks = []
    all_visibilities = []
    total_model_time = 0.0
    global_frame_offset = 0  # 新增：跟踪全局帧偏移

    for video_chunk in read_video_with_ffmpeg_chunked(args.video_path, chunk_size=args.chunk_size):
        pair_start_time = time.time()
        print(f"\nProcessing chunk {chunk_idx}, frames: {video_chunk.shape[0]}")
        
        print("Converting chunk to PyTorch tensor")
        video = torch.from_numpy(video_chunk).permute(0, 3, 1, 2)[None].float()
        video = video.to(DEFAULT_DEVICE)
        print(f"Chunk tensor created, shape: {video.shape}, device: {video.device}")

        # 新增：绘制查询点位置
        plot_query_points(queries, video.shape, save_dir, chunk_idx)

        # 查询 GPU 占用
        used_mb_before, total_mb = get_gpu_memory(gpu_handle)
        if used_mb_before is not None:
            print(
                f"Before inference (chunk {chunk_idx}): "
                f"GPU memory used: {used_mb_before:.2f} MB / {total_mb:.2f} MB "
                f"({used_mb_before / total_mb * 100:.2f}%)"
            )

        # 模型推理，传递 queries
        print("Running model inference with queries")
        model_start_time = time.time()
        with torch.no_grad():  # 新增：禁用梯度计算以优化内存
            pred_tracks, pred_visibility = model(
                video,
                queries=queries[None],  # 修改：传递查询点
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame if chunk_idx == 0 else 0,
                backward_tracking=args.backward_tracking,
            )
        model_time = time.time() - model_start_time
        total_model_time += model_time

        # 查询推理后 GPU 占用
        used_mb_after, _ = get_gpu_memory(gpu_handle)
        if used_mb_after is not None and used_mb_before is not None:
            memory_diff = used_mb_after - used_mb_before
            print(
                f"After inference (chunk {chunk_idx}): "
                f"GPU memory used: {used_mb_after:.2f} MB / {total_mb:.2f} MB "
                f"({used_mb_after / total_mb * 100:.2f}%), "
                f"Change: {'+' if memory_diff >= 0 else ''}{memory_diff:.2f} MB"
            )
        print(f"Chunk {chunk_idx} inference completed, tracks shape: {pred_tracks.shape}, model time: {model_time:.2f} seconds")

        # 新增：记录跟踪点位置
        log_track_positions(pred_tracks, chunk_idx, global_frame_offset)
        global_frame_offset += video_chunk.shape[0]  # 更新全局帧偏移

        # 存储结果
        all_tracks.append(pred_tracks.cpu())
        all_visibilities.append(pred_visibility.cpu())
        print(f"Chunk {chunk_idx} results stored, total chunks: {len(all_tracks)}")

        # 可视化当前块
        print("Visualizing chunk")
        vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=6, mode='cool', tracks_leave_trace=-1)
        chunk_save_path = os.path.join(save_dir, f"{seq_name}_chunk_{chunk_idx}.mp4")
        vis.visualize(
            video,
            pred_tracks,
            pred_visibility,
            query_frame=0 if args.backward_tracking else args.grid_query_frame if chunk_idx == 0 else 0,
            filename=chunk_save_path
        )
        print(f"Chunk {chunk_idx} visualized and saved to {chunk_save_path}")
        
        pair_time = time.time() - pair_start_time
        print(f"(Chunk total time: {pair_time:.2f} seconds)")

        # 清理内存
        print("Cleaning up memory")
        del video, pred_tracks, pred_visibility
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        chunk_idx += 1

    # 合并结果
    if all_tracks and all_visibilities:
        print("\nMerging all chunk results")
        final_tracks = torch.cat(all_tracks, dim=1)
        final_visibility = torch.cat(all_visibilities, dim=1)
        print(f"Final tracks shape: {final_tracks.shape}, visibility shape: {final_visibility.shape}")
        
        # 新增：可视化合并后的查询点跟踪结果
        print("Visualizing final tracks with queries")
        vis = Visualizer(save_dir=save_dir, linewidth=6, mode='cool', tracks_leave_trace=-1)
        final_save_path = os.path.join(save_dir, f"{seq_name}_queries.mp4")
        vis.visualize(
            video=torch.cat([torch.from_numpy(chunk).permute(0, 3, 1, 2)[None].float().to(DEFAULT_DEVICE) for chunk in read_video_with_ffmpeg_chunked(args.video_path, chunk_size=args.chunk_size)], dim=1),
            tracks=final_tracks.to(DEFAULT_DEVICE),
            visibility=final_visibility.to(DEFAULT_DEVICE),
            filename=final_save_path
        )
        print(f"Final tracks visualized and saved to {final_save_path}")

    else:
        print("No chunks processed, skipping merge")

    # 关闭 NVML
    if gpu_handle is not None:
        pynvml.nvmlShutdown()
        print("NVML shutdown completed")

    # 计算并打印程序总运行时间和模型处理总时间
    total_time = time.time() - start_time
    print(f"\nProcessing completed!")
    print(f"Total program runtime: {total_time:.2f} seconds")
    print(f"Total model inference time: {total_model_time:.2f} seconds")
    print('-' * 10 + " Section ends here " + '-' * 12 + "\n")

log_file.close()