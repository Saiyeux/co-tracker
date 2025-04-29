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
import uuid

# 日志文件以时间戳命名
log_filename = f"output_{time.strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_filename, "w")
sys.stdout = log_file

# 记录整个过程的开始时间
start_time = time.time()

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# 区域框线颜色，BGR 格式
REGION_COLORS = [
    (0, 0, 255),    # 区域 0: 红色
    (0, 255, 0),    # 区域 1: 绿色
    (255, 0, 0),    # 区域 2: 蓝色
    (255, 255, 0)   # 区域 3: 黄色
]

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
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
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

# 判断点是否在四边形内（使用射线法）
def point_in_polygon(x, y, poly):
    """
    poly: 四边形顶点 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    返回：True 如果点 (x, y) 在四边形内（包括边界），否则 False
    """
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        if ((poly[i][1] > y) != (poly[j][1] > y)) and \
           (x < (poly[j][0] - poly[i][0]) * (y - poly[i][1]) / (poly[j][1] - poly[i][1]) + poly[i][0]):
            inside = not inside
        j = i
    return inside

# 生成网格点，判断哪些点在四边形内
def generate_grid_points(point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y, spacing, frame_number=0, width=1920, height=1080, region_idx=0):
    # 定义四边形顶点：左上、右上、右下、左下
    poly = [(point1_x, point1_y), (point2_x, point2_y), (point3_x, point3_y), (point4_x, point4_y)]
    
    # 计算四边形的边界框以生成网格
    min_x = min(point1_x, point2_x, point3_x, point4_x)
    max_x = max(point1_x, point2_x, point3_x, point4_x)
    min_y = min(point1_y, point2_y, point3_y, point4_y)
    max_y = max(point1_y, point2_y, point3_y, point4_y)
    
    x_coords = np.arange(min_x, max_x + spacing, spacing)
    y_coords = np.arange(min_y, max_y + spacing, spacing)
    
    queries = []
    for y in y_coords:
        for x in x_coords:
            if 0 <= x < width and 0 <= y < height:
                # 判断点是否在四边形内
                if point_in_polygon(x, y, poly):
                    queries.append([float(frame_number), float(x), float(y)])
    
    if not queries:
        raise ValueError(f"No valid grid points within four-point polygon for region {region_idx}")
    
    queries = torch.tensor(queries)
    
    # 记录初始四边形顶点
    poly_coords = {
        'top_left': (point1_x, point1_y),
        'top_right': (point2_x, point2_y),
        'bottom_right': (point3_x, point3_y),
        'bottom_left': (point4_x, point4_y)
    }

    # 记录四边形的坐标
    box_coords = {
        'top_left': (point1_x, point1_y),
        'top_right': (point2_x, point2_y),
        'bottom_right': (point3_x, point3_y),
        'bottom_left': (point4_x, point4_y)
    }
    
    print(f"Generated {len(queries)} grid points for region {region_idx} with spacing {spacing:.2f}, "
          f"Polygon coords: top_left=({point1_x}, {point1_y}), top_right=({point2_x}, {point2_y}), "
          f"bottom_right=({point3_x}, {point3_y}), bottom_left=({point4_x}, {point4_y})")
    return queries, box_coords

# 从上一 chunk 的结束位置生成新的 queries
def generate_queries_from_last_positions(region, region_idx, frame_number=0, width=1920, height=1080):
    if 'last_positions' not in region:
        raise ValueError(f"No last_positions found for region {region_idx}. Ensure previous chunk was processed.")
    
    last_positions = region['last_positions']
    num_points = len(last_positions)
    queries = np.zeros((num_points, 3))
    queries[:, 0] = frame_number
    queries[:, 1:3] = last_positions
    
    # 验证坐标有效性
    valid_mask = (0 <= queries[:, 1]) & (queries[:, 1] < width) & (0 <= queries[:, 2]) & (queries[:, 2] < height)
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        print(f"Warning: {invalid_count} points in region {region_idx} have invalid coordinates outside [0, {width}]x[0, {height}]")
        queries[~valid_mask, 1:3] = region['queries'][~valid_mask, 1:3].cpu().numpy()
    
    queries = torch.tensor(queries, dtype=torch.float32)
    
    # 日志：记录新 queries 的坐标范围
    min_x, max_x = queries[:, 1].min().item(), queries[:, 1].max().item()
    min_y, max_y = queries[:, 2].min().item(), queries[:, 2].max().item()
    print(f"Chunk {chunk_idx}, Region {region_idx}: New queries range x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")
    
    return queries

# 更新区域内点的平均位置，计算新的四边形坐标
def log_movement_matrix(pred_tracks, pred_visibility, chunk_idx, global_frame_offset, region_idx=0):
    print(f"\nProcessing movement for chunk {chunk_idx}, region {region_idx}")
    num_frames = pred_tracks.shape[1]
    num_points = pred_tracks.shape[2]
    
    # 存储每帧的平均位置
    frame_positions = []
    
    for frame_idx in range(num_frames):
        global_frame = global_frame_offset + frame_idx
        valid_points = []
        for point_idx in range(num_points):
            if pred_visibility[0, frame_idx, point_idx].item():
                x, y = pred_tracks[0, frame_idx, point_idx, 0].item(), pred_tracks[0, frame_idx, point_idx, 1].item()
                valid_points.append([x, y])
        
        if valid_points:
            valid_points = np.array(valid_points)
            mean_x = np.mean(valid_points[:, 0])
            mean_y = np.mean(valid_points[:, 1])
            frame_positions.append({'frame': global_frame, 'mean_x': mean_x, 'mean_y': mean_y})
        else:
            print(f"Warning: No visible points in frame {global_frame}, region {region_idx}")
            frame_positions.append({'frame': global_frame, 'mean_x': None, 'mean_y': None})
    
    # 更新 last_positions
    last_positions = np.zeros((num_points, 2))
    for i in range(num_points):
        if pred_visibility[0, -1, i].item():
            last_positions[i] = pred_tracks[0, -1, i, :].cpu().numpy()
        else:
            last_positions[i] = (region['last_positions'][i] if 'last_positions' in region 
                                 else region['queries'][i, 1:3].cpu().numpy())
    
    print(f"Chunk {chunk_idx}, Region {region_idx}: Updated {len(last_positions)} points' last positions")
    return frame_positions, last_positions

# 绘制四边形区域框线，动态更新位置
def custom_visualize(video, tracks, visibility, regions, save_path, fps=30):
    print(f"Generating visualization video at {save_path}")
    height, width = video.shape[3], video.shape[4]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    num_frames = min(video.shape[1], tracks.shape[1])
    
    print(f"Visualizing {len(regions)} regions with quadrilateral boxes")
    
    for t in range(num_frames):
        frame = video[0, t].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        for region_idx, region in enumerate(regions):
            start_idx = region['point_offset']
            end_idx = start_idx + region['queries'].shape[0]
            region_tracks = tracks[:, :, start_idx:end_idx]
            region_visibility = visibility[:, :, start_idx:end_idx]
            
            # 获取当前帧的平均位置
            frame_positions = region.get('frame_positions', [])
            if t < len(frame_positions) and frame_positions[t]['mean_x'] is not None:
                mean_x = frame_positions[t]['mean_x']
                mean_y = frame_positions[t]['mean_y']
                
                # 计算初始四边形的边界框尺寸
                init_coords = [
                    region['box_coords']['top_left'],
                    region['box_coords']['top_right'],
                    region['box_coords']['bottom_right'],
                    region['box_coords']['bottom_left']
                ]
                init_xs = [p[0] for p in init_coords]
                init_ys = [p[1] for p in init_coords]
                box_width = max(init_xs) - min(init_xs)
                box_height = max(init_ys) - min(init_ys)
                
                # 计算偏移量，保持四边形形状
                dx = mean_x - (min(init_xs) + box_width / 2)
                dy = mean_y - (min(init_ys) + box_height / 2)
                
                # 更新四边形顶点
                new_coords = [
                    (max(0, min(init_coords[0][0] + dx, width - 1)), max(0, min(init_coords[0][1] + dy, height - 1))),  # top_left
                    (max(0, min(init_coords[1][0] + dx, width - 1)), max(0, min(init_coords[1][1] + dy, height - 1))),  # top_right
                    (max(0, min(init_coords[2][0] + dx, width - 1)), max(0, min(init_coords[2][1] + dy, height - 1))),  # bottom_right
                    (max(0, min(init_coords[3][0] + dx, width - 1)), max(0, min(init_coords[3][1] + dy, height - 1)))   # bottom_left
                ]
                
                # 绘制四边形
                color = REGION_COLORS[region_idx % len(REGION_COLORS)]
                pts = np.array(new_coords, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, color, thickness=2)
                print(f"Frame {t}, Region {region_idx}: Quadrilateral at {new_coords}")
            else:
                # 如果没有有效点，绘制初始四边形
                init_coords = [
                    region['box_coords']['top_left'],
                    region['box_coords']['top_right'],
                    region['box_coords']['bottom_right'],
                    region['box_coords']['bottom_left']
                ]
                color = REGION_COLORS[region_idx % len(REGION_COLORS)]
                pts = np.array(init_coords, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, color, thickness=2)
                print(f"Frame {t}, Region {region_idx}: No valid points, using initial quadrilateral")
        
        out.write(frame)
        del frame
    
    out.release()
    print(f"Visualization video saved to {save_path}")

if __name__ == "__main__":
    print("\n" + '-' * 10 + " Section starts here " + '-' * 10)
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="/home/surgicalai/Data/test5.1-1.mp4", help="path to a video")
    parser.add_argument("--points", type=float, nargs='+', default=[975, 875, 1100, 900, 1050, 1025, 900, 1000], 
                        help="List of points as [x1, y1, x2, y2, x3, y3, x4, y4, ...] for multiple quadrilateral regions")
    parser.add_argument("--spacing", type=float, default=10, help="Spacing between grid points")
    parser.add_argument("--checkpoint", default=None, help="CoTracker model parameters")
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument("--grid_query_frame", type=int, default=0, help="Compute dense and grid tracks starting from this frame")
    parser.add_argument("--backward_tracking", action="store_true", help="Compute tracks in both directions")
    parser.add_argument("--use_v2_model", action="store_true", help="Use CoTracker2 instead of CoTracker++")
    parser.add_argument("--offline", action="store_true", help="Use the offline model")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of frames per chunk")
    parser.add_argument("--save_dir", default="/home/surgicalai/Data/output/co-tracker", help="Directory to save output videos")

    args = parser.parse_args()
    print(f"Arguments parsed: video_path={args.video_path}, chunk_size={args.chunk_size}, save_dir={args.save_dir}")
    print(f"Regions: {len(args.points)//8} regions, Points: {args.points}, Spacing={args.spacing}")

    # 验证视频文件
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")

    # 验证点输入
    if len(args.points) % 8 != 0:
        raise ValueError("Number of points must be a multiple of 8 (x1, y1, x2, y2, x3, y3, x4, y4 per region)")

    # 获取视频尺寸
    probe = ffmpeg.probe(args.video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    # 初始化区域
    regions = []
    total_points = 0
    for i in range(0, len(args.points), 8):
        point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y = args.points[i:i+8]
        queries, box_coords = generate_grid_points(
            point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y,
            args.spacing, width=width, height=height, region_idx=i//8
        )
        regions.append({
            'queries': queries,
            'box_coords': box_coords,
            'point_offset': total_points,
            'colors': REGION_COLORS[i//8 % len(REGION_COLORS)]
        })
        total_points += queries.shape[0]
    
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
            
            # 更新 queries
            if chunk_idx > 0:
                print(f"Updating queries for chunk {chunk_idx} using last positions")
                updated_regions = []
                total_points = 0
                for i, region in enumerate(regions):
                    queries = generate_queries_from_last_positions(
                        region, region_idx=i, frame_number=0, width=width, height=height
                    )
                    updated_regions.append({
                        'queries': queries,
                        'box_coords': region['box_coords'],
                        'point_offset': total_points,
                        'colors': region['colors']
                    })
                    total_points += queries.shape[0]
                regions = updated_regions
            
            # 合并所有区域的查询点
            all_queries = torch.cat([r['queries'] for r in regions], dim=0)
            if torch.cuda.is_available():
                all_queries = all_queries.cuda()
            print(f"Queries defined, shape: {all_queries.shape}, device: {all_queries.device}")
            print(f"Total grid points: {total_points}, Regions: {len(regions)}")
            
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
                    queries=all_queries[None],
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

            # 为每个区域处理运动，更新 frame_positions 和 last_positions
            for region_idx, region in enumerate(regions):
                start_idx = region['point_offset']
                end_idx = start_idx + region['queries'].shape[0]
                region_tracks = pred_tracks[:, :, start_idx:end_idx]
                region_visibility = pred_visibility[:, :, start_idx:end_idx]
                
                frame_positions, last_positions = log_movement_matrix(
                    region_tracks, region_visibility, chunk_idx, global_frame_offset, region_idx
                )
                region['frame_positions'] = frame_positions
                region['last_positions'] = last_positions
            
            global_frame_offset += video_chunk.shape[0]

            # 可视化
            print("Visualizing chunk with quadrilateral region boxes")
            chunk_save_path = os.path.join(save_dir, f"{seq_name}_chunk_{chunk_idx}.mp4")
            custom_visualize(video, pred_tracks, pred_visibility, regions, chunk_save_path)
            
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