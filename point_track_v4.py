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
import time
import sys
import cv2

# 日志文件以时间戳命名
log_filename = f"output_{time.strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_filename, "w")
sys.stdout = log_file

# 记录整个过程的开始时间
start_time = time.time()

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# 初始化 NVML
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

# 分块读取视频
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
        
        try:
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
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None  # 释放显存
        finally:
            process.stdout.close()
            process.wait()
            print("Video reading completed")
    
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise
    except Exception as e:
        print(f"Error reading video: {e}")
        raise

# 生成网格点并记录邻居关系
def generate_grid_points(point1_x, point1_y, point2_x, point2_y, spacing, frame_number=0, width=1920, height=1080, max_points=100):
    min_x = min(point1_x, point2_x)
    max_x = max(point1_x, point2_x)
    min_y = min(point1_y, point2_y)
    max_y = max(point1_y, point2_y)
    
    # 动态调整 spacing 以限制点数
    x_points = int((max_x - min_x) / spacing) + 1
    y_points = int((max_y - min_y) / spacing) + 1
    if x_points * y_points > max_points:
        factor = np.sqrt(max_points / (x_points * y_points))
        spacing /= factor
        print(f"Adjusted spacing to {spacing:.2f} to limit points to ~{max_points}")
    
    x_coords = np.arange(min_x, max_x + spacing, spacing)
    y_coords = np.arange(min_y, max_y + spacing, spacing)
    
    queries = []
    grid_shape = (len(y_coords), len(x_coords))
    point_to_index = {}
    index = 0
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            if 0 <= x < width and 0 <= y < height:
                queries.append([float(frame_number), float(x), float(y)])
                point_to_index[(i, j)] = index
                index += 1
    
    queries = torch.tensor(queries)
    if queries.shape[0] == 0:
        raise ValueError("No valid grid points within video boundaries")
    
    # 计算初始邻居关系和距离
    neighbors = {}
    initial_distances = {}
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if (i, j) not in point_to_index:
                continue
            idx = point_to_index[(i, j)]
            neighbor_list = []
            # 上
            if i > 0 and (i-1, j) in point_to_index:
                neighbor_idx = point_to_index[(i-1, j)]
                neighbor_list.append(neighbor_idx)
                dist = np.sqrt((queries[idx, 1] - queries[neighbor_idx, 1])**2 + 
                              (queries[idx, 2] - queries[neighbor_idx, 2])**2)
                initial_distances[(idx, neighbor_idx)] = dist.item()
            # 下
            if i < grid_shape[0] - 1 and (i+1, j) in point_to_index:
                neighbor_idx = point_to_index[(i+1, j)]
                neighbor_list.append(neighbor_idx)
                dist = np.sqrt((queries[idx, 1] - queries[neighbor_idx, 1])**2 + 
                              (queries[idx, 2] - queries[neighbor_idx, 2])**2)
                initial_distances[(idx, neighbor_idx)] = dist.item()
            # 左
            if j > 0 and (i, j-1) in point_to_index:
                neighbor_idx = point_to_index[(i, j-1)]
                neighbor_list.append(neighbor_idx)
                dist = np.sqrt((queries[idx, 1] - queries[neighbor_idx, 1])**2 + 
                              (queries[idx, 2] - queries[neighbor_idx, 2])**2)
                initial_distances[(idx, neighbor_idx)] = dist.item()
            # 右
            if j < grid_shape[1] - 1 and (i, j+1) in point_to_index:
                neighbor_idx = point_to_index[(i, j+1)]
                neighbor_list.append(neighbor_idx)
                dist = np.sqrt((queries[idx, 1] - queries[neighbor_idx, 1])**2 + 
                              (queries[idx, 2] - queries[neighbor_idx, 2])**2)
                initial_distances[(idx, neighbor_idx)] = dist.item()
            neighbors[idx] = neighbor_list
    
    print(f"Generated {len(queries)} grid points with spacing {spacing:.2f}, grid shape: {grid_shape}")
    return queries, grid_shape, neighbors, initial_distances, point_to_index

# 计算间距变化并记录移动点
def log_movement_matrix(pred_tracks, chunk_idx, global_frame_offset, grid_shape, neighbors, initial_distances, point_to_index):
    print(f"\nProcessing movement for chunk {chunk_idx}")
    num_frames = pred_tracks.shape[1]
    num_points = pred_tracks.shape[2]
    
    # 初始化变化矩阵
    movement_matrix = np.zeros(grid_shape, dtype=int)
    moved_points = set()
    moved_info = {}  # 按点汇总首次移动
    
    for frame_idx in range(num_frames):
        global_frame = global_frame_offset + frame_idx
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                if (i, j) not in point_to_index:
                    continue
                point_idx = point_to_index[(i, j)]
                if point_idx in moved_info:  # 跳过已记录的点
                    movement_matrix[i, j] = 1
                    moved_points.add(point_idx)
                    continue
                x1, y1 = pred_tracks[0, frame_idx, point_idx, 0].item(), pred_tracks[0, frame_idx, point_idx, 1].item()
                
                for neighbor_idx in neighbors[point_idx]:
                    x2, y2 = pred_tracks[0, frame_idx, neighbor_idx, 0].item(), pred_tracks[0, frame_idx, neighbor_idx, 1].item()
                    current_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    initial_dist = initial_distances.get((point_idx, neighbor_idx), current_dist)
                    
                    if current_dist > initial_dist + 0.1:
                        movement_matrix[i, j] = 1
                        moved_points.add(point_idx)
                        moved_info[point_idx] = {
                            'frame': global_frame,
                            'x': x1,
                            'y': y1,
                            'neighbor_idx': neighbor_idx,
                            'current_dist': current_dist,
                            'initial_dist': initial_dist
                        }
                        break  # 记录一次移动后跳出
    
    # 打印变化矩阵
    print(f"\nMovement matrix for chunk {chunk_idx} (1=moved, 0=unmoved):")
    for row in movement_matrix:
        print(' '.join(map(str, row)))
    
    # 汇总移动点信息
    if moved_info:
        print(f"\nMoved points for chunk {chunk_idx}:")
        for point_idx, move in moved_info.items():
            print(f"Point {point_idx}: Frame {move['frame']}: (x={move['x']:.2f}, y={move['y']:.2f}), "
                  f"Distance to neighbor {move['neighbor_idx']}: {move['current_dist']:.2f} "
                  f"(initial: {move['initial_dist']:.2f})")
    
    return movement_matrix, moved_points

# 自定义视频可视化
def custom_visualize(video, tracks, visibility, moved_points, save_path, fps=30):
    print(f"Generating custom video at {save_path}")
    height, width = video.shape[3], video.shape[4]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    num_frames = min(video.shape[1], tracks.shape[1])  # 确保帧数一致
    num_points = tracks.shape[2]
    
    for t in range(num_frames):
        frame = video[0, t].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        for p in range(num_points):
            if visibility[0, t, p].item():
                x, y = int(tracks[0, t, p, 0].item()), int(tracks[0, t, p, 1].item())
                color = (0, 0, 255) if p in moved_points else (255, 0, 0)  # 红=移动，蓝=未移动
                cv2.circle(frame, (x, y), 5, color, -1)
        
        out.write(frame)
        del frame  # 释放帧内存
    
    out.release()
    print(f"Custom video saved to {save_path}")

if __name__ == "__main__":
    print("\n" + '-' * 10 + " Section starts here " + '-' * 10)
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="/home/surgicalai/Data/test5.mp4", help="path to a video")
    parser.add_argument("--point1_x", type=float, default=850, help="X coordinate of first point")
    parser.add_argument("--point1_y", type=float, default=450, help="Y coordinate of first point")
    parser.add_argument("--point2_x", type=float, default=1050, help="X coordinate of second point")
    parser.add_argument("--point2_y", type=float, default=850, help="Y coordinate of second point")
    parser.add_argument("--spacing", type=float, default=50, help="Spacing between grid points")
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
    print(f"Grid points: Point1=({args.point1_x}, {args.point1_y}), Point2=({args.point2_x}, {args.point2_y}), Spacing={args.spacing}")

    # 验证视频文件
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")

    # 获取视频尺寸
    probe = ffmpeg.probe(args.video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    # 生成网格点和邻居关系
    queries, grid_shape, neighbors, initial_distances, point_to_index = generate_grid_points(
        args.point1_x, args.point1_y,
        args.point2_x, args.point2_y,
        args.spacing,
        width=width,
        height=height,
        max_points=100
    )
    if torch.cuda.is_available():
        queries = queries.cuda()
    print(f"Queries defined,禁止: shape: {queries.shape}, device: {queries.device}")
    print(f"Initial grid points: {queries.shape[0]}, Grid shape: {grid_shape}")

    # 初始化 NVML
    gpu_handle = initialize_nvml()

    # 初始化模型
    print("Initializing CoTracker model")
    try:
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
    except Exception as e:
        print(f"Model initialization failed: {e}")
        raise

    # 分块处理视频
    seq_name = args.video_path.split("/")[-1].split(".")[0]
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory prepared: {save_dir}")
    
    chunk_idx = 0
    total_model_time = 0.0
    global_frame_offset = 0

    try:
        for video_chunk in read_video_with_ffmpeg_chunked(args.video_path, chunk_size=args.chunk_size):
            pair_start_time = time.time()
            print(f"\nProcessing chunk {chunk_idx}, frames: {video_chunk.shape[0]}")
            
            # 转换视频块为张量
            print("Converting chunk to PyTorch tensor")
            video = torch.from_numpy(video_chunk).permute(0, 3, 1, 2)[None].float().to(DEFAULT_DEVICE)
            print(f"Chunk tensor created, shape: {video.shape}, device: {video.device}")

            # 查询 GPU 占用
            used_mb_before, total_mb = get_gpu_memory(gpu_handle)
            if used_mb_before is not None:
                print(
                    f"Before inference (chunk {chunk_idx}): "
                    f"GPU memory used: {used_mb_before:.2f} MB / {total_mb:.2f} MB "
                    f"({used_mb_before / total_mb * 100:.2f}%)"
                )

            # 模型推理
            print("Running model inference with grid points")
            model_start_time = time.time()
            with torch.no_grad():
                pred_tracks, pred_visibility = model(
                    video,
                    queries=queries[None],
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

            # 计算间距变化并记录
            movement_matrix, moved_points = log_movement_matrix(
                pred_tracks, chunk_idx, global_frame_offset, grid_shape, neighbors, 
                initial_distances, point_to_index
            )
            global_frame_offset += video_chunk.shape[0]

            # 自定义可视化
            print("Visualizing chunk with custom colors")
            chunk_save_path = os.path.join(save_dir, f"{seq_name}_chunk_{chunk_idx}.mp4")
            custom_visualize(video, pred_tracks, pred_visibility, moved_points, chunk_save_path)
            
            pair_time = time.time() - pair_start_time
            print(f"(Chunk total time: {pair_time:.2f} seconds)")

            # 清理内存
            print("Cleaning up memory")
            del video, pred_tracks, pred_visibility, video_chunk
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            chunk_idx += 1

    except Exception as e:
        print(f"Processing failed: {e}")
        raise
    finally:
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