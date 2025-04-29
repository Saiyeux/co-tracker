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
import json

# 日志文件以时间戳命名
log_filename = f"output_{time.strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_filename, "w")
sys.stdout = log_file

# 记录整个过程的开始时间
start_time = time.time()

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# 区域颜色对：(未移动色, 移动色)，BGR 格式
REGION_COLORS = [
    ((255, 0, 0), (0, 0, 255)),    # 蓝色未移动，红色移动
    ((0, 255, 0), (0, 255, 255)),   # 绿色未移动，黄色移动
    ((255, 255, 0), (255, 0, 255)), # 青色未移动，紫色移动
    ((128, 128, 128), (255, 128, 0)) # 灰色未移动，橙色移动
]

# 颜色名称映射，用于日志
COLOR_NAMES = {
    (255, 0, 0): 'Blue',
    (0, 0, 255): 'Red',
    (0, 255, 0): 'Green',
    (0, 255, 255): 'Yellow',
    (255, 255, 0): 'Cyan',
    (255, 0, 255): 'Purple',
    (128, 128, 128): 'Gray',
    (255, 128, 0): 'Orange'
}

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

# 从 JSON 文件加载跟踪点
def load_points_from_json(json_path, region_id, width, height):
    """
    从 JSON 文件中加载指定 region_id 的跟踪点。
    
    Args:
        json_path: Path to the JSON file
        region_id: ID of the region to load points from
        width, height: Video dimensions for validation
    
    Returns:
        points: List of [x1, y1, x2, y2, ...] coordinates
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'regions' not in data:
            raise ValueError("JSON file must contain 'regions' key")
        
        for region in data['regions']:
            if region.get('region_id') == region_id:
                points = region.get('tracking_points', [])
                if not points:
                    raise ValueError(f"No tracking points found for region_id {region_id}")
                
                # 展平点坐标为 [x1, y1, x2, y2, ...]
                flat_points = []
                for x, y in points:
                    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                        raise ValueError(f"Invalid point coordinates: ({x}, {y})")
                    flat_points.extend([float(x), float(y)])
                
                print(f"Loaded {len(flat_points) // 2} points from region {region_id}")
                return flat_points
        
        raise ValueError(f"Region ID {region_id} not found in JSON file")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {json_path}")
    except Exception as e:
        raise ValueError(f"Error loading JSON file: {e}")

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

# 生成跟踪点和邻居关系
def generate_tracking_points(points, frame_number=0, width=1920, height=1080, region_idx=0, neighbor_distance=20):
    """
    根据输入的(x, y)坐标生成跟踪点，并基于距离阈值确定邻居关系。
    
    Args:
        points: List of [x1, y1, x2, y2, ...] coordinates
        frame_number: Starting frame number
        width, height: Video dimensions
        region_idx: Region index for logging
        neighbor_distance: Maximum distance to consider two points as neighbors
    
    Returns:
        queries: Tensor of shape (N, 3) with [frame_number, x, y]
        neighbors: Dict mapping point index to list of neighbor indices
        initial_distances: Dict of (point_idx, neighbor_idx) to initial distance
        point_to_index: Dict mapping point index to its position
    """
    if len(points) % 2 != 0:
        raise ValueError("Points must be provided as pairs of (x, y) coordinates")
    
    num_points = len(points) // 2
    queries = []
    point_to_index = {}
    
    # 生成查询点
    for i in range(0, len(points), 2):
        x, y = points[i], points[i + 1]
        if 0 <= x < width and 0 <= y < height:
            queries.append([float(frame_number), float(x), float(y)])
            point_to_index[i // 2] = i // 2
        else:
            print(f"Warning: Point {i // 2} at ({x}, {y}) is outside video boundaries [0, {width}]x[0, {height}]")
    
    if not queries:
        raise ValueError(f"No valid points within video boundaries for region {region_idx}")
    
    queries = torch.tensor(queries, dtype=torch.float32)
    print(f"Generated {len(queries)} tracking points for region {region_idx}")
    
    # 计算邻居关系和初始距离
    neighbors = {i: [] for i in range(len(queries))}
    initial_distances = {}
    
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            x1, y1 = queries[i, 1], queries[i, 2]
            x2, y2 = queries[j, 1], queries[j, 2]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist <= neighbor_distance:
                neighbors[i].append(j)
                neighbors[j].append(i)
                initial_distances[(i, j)] = dist.item()
                initial_distances[(j, i)] = dist.item()
    
    print(f"Computed neighbors for {len(queries)} points, max neighbor distance: {neighbor_distance}px")
    return queries, neighbors, initial_distances, point_to_index

# 从上一 chunk 的结束位置生成新的 queries
def generate_queries_from_last_positions(region, region_idx, frame_number=0, width=1920, height=1080):
    if 'last_positions' not in region:
        raise ValueError(f"No last_positions found for region {region_idx}. Ensure previous chunk was processed.")
    
    last_positions = region['last_positions']
    num_points = len(last_positions)
    queries = np.zeros((num_points, 3))
    queries[:, 0] = frame_number  # 设置 frame_number 为 0
    queries[:, 1:3] = last_positions  # 使用上一 chunk 结束时的 x, y 坐标
    
    # 验证坐标有效性
    valid_mask = (0 <= queries[:, 1]) & (queries[:, 1] < width) & (0 <= queries[:, 2]) & (queries[:, 2] < height)
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        print(f"Warning: {invalid_count} points in region {region_idx} have invalid coordinates outside [0, {width}]x[0, {height}]")
        queries[~valid_mask, 1:3] = region['queries'][~valid_mask, 1:3].cpu().numpy()  # 回退到初始位置
    
    queries = torch.tensor(queries, dtype=torch.float32)
    
    # 日志：记录新 queries 的坐标范围
    min_x, max_x = queries[:, 1].min().item(), queries[:, 1].max().item()
    min_y, max_y = queries[:, 2].min().item(), queries[:, 2].max().item()
    print(f"Chunk {chunk_idx}, Region {region_idx}: New queries range x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")
    
    return queries

# 计算间距变化并记录移动点，更新 last_positions
def log_movement_matrix(pred_tracks, pred_visibility, chunk_idx, global_frame_offset, neighbors, initial_distances, point_to_index, threshold, region_idx=0):
    print(f"\nProcessing movement for chunk {chunk_idx}, region {region_idx}")
    num_frames = pred_tracks.shape[1]
    num_points = pred_tracks.shape[2]
    
    # 初始化变化矩阵
    movement_matrix = np.zeros(num_points, dtype=int)
    moved_points = set()
    moved_info = {}
    distance_changes = []
    moved_connections = []
    
    for frame_idx in range(num_frames):
        global_frame = global_frame_offset + frame_idx
        for point_idx in range(num_points):
            if point_idx in moved_info:
                movement_matrix[point_idx] = 1
                moved_points.add(point_idx)
                continue
            x1, y1 = pred_tracks[0, frame_idx, point_idx, 0].item(), pred_tracks[0, frame_idx, point_idx, 1].item()
            
            max_relative_change = 0.0
            max_neighbor_idx = None
            current_dist = None
            initial_dist = None
            for neighbor_idx in neighbors[point_idx]:
                x2, y2 = pred_tracks[0, frame_idx, neighbor_idx, 0].item(), pred_tracks[0, frame_idx, neighbor_idx, 1].item()
                current_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                initial_dist = initial_distances.get((point_idx, neighbor_idx), current_dist)
                relative_change = abs(current_dist - initial_dist)
                
                if relative_change > threshold:
                    moved_connections.append({
                        'frame': global_frame,
                        'point_idx': point_idx,
                        'neighbor_idx': neighbor_idx,
                        'relative_change': relative_change
                    })
                
                if relative_change > max_relative_change:
                    max_relative_change = relative_change
                    max_neighbor_idx = neighbor_idx
                    current_dist = current_dist
                    initial_dist = initial_dist
            
            distance_changes.append(max_relative_change)
            if max_relative_change > threshold:
                movement_matrix[point_idx] = 1
                moved_points.add(point_idx)
                moved_info[point_idx] = {
                    'frame': global_frame,
                    'x': x1,
                    'y': y1,
                    'neighbor_idx': max_neighbor_idx,
                    'relative_change': max_relative_change,
                    'current_dist': current_dist,
                    'initial_dist': initial_dist
                }
    
    # 汇总移动点信息
    print(f"\nMoved points count for region {region_idx}: {len(moved_info)}")
    if moved_info:
        print(f"Moved points details for region {region_idx}:")
        for point_idx, move in moved_info.items():
            print(f"Point {point_idx}: Frame {move['frame']}: (x={move['x']:.2f}, y={move['y']:.2f}), "
                  f"Relative distance change to neighbor {move['neighbor_idx']}: {move['relative_change']:.4f} px "
                  f"(current: {move['current_dist']:.2f}, initial: {move['initial_dist']:.2f})")
    else:
        print(f"Warning: No moved points detected in region {region_idx}. Consider lowering the threshold (currently {threshold} px).")
    
    # 汇总超阈值连接信息
    print(f"\nMoved connections count for region {region_idx}: {len(moved_connections)}")
    if moved_connections:
        print(f"Moved connections details for region {region_idx}:")
        for conn in moved_connections:
            print(f"Frame {conn['frame']}: Point {conn['point_idx']} to Neighbor {conn['neighbor_idx']}, "
                  f"Relative change: {conn['relative_change']:.4f} px")
    
    # 调试：打印距离变化统计
    if distance_changes:
        print(f"Distance change stats for region {region_idx}: min={min(distance_changes):.4f}, max={max(distance_changes):.4f}, "
              f"mean={np.mean(distance_changes):.4f}, std={np.std(distance_changes):.4f}")
    
    # 更新 last_positions
    last_positions = np.zeros((num_points, 2))
    for i in range(num_points):
        if pred_visibility[0, -1, i].item():
            last_positions[i] = pred_tracks[0, -1, i, :].cpu().numpy()
        else:
            last_positions[i] = (region['last_positions'][i] if 'last_positions' in region 
                                 else region['queries'][i, 1:3].cpu().numpy())
    print(f"Chunk {chunk_idx}, Region {region_idx}: Updated {len(last_positions)} points' last positions")
    
    return movement_matrix, moved_points, moved_connections, initial_distances, last_positions

# 自定义视频可视化
def custom_visualize(video, tracks, visibility, regions, save_path, threshold=0.5, fps=30):
    print(f"Generating custom video at {save_path}")
    height, width = video.shape[3], video.shape[4]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    num_frames = min(video.shape[1], tracks.shape[1])
    total_points = tracks.shape[2]
    
    # 合并所有区域的移动点和连接
    all_moved_points = set()
    all_moved_connections = []
    for region in regions:
        all_moved_points.update(region['moved_points'])
        all_moved_connections.extend(region['moved_connections'])
    
    print(f"Visualizing {total_points} points across {len(regions)} regions, {len(all_moved_points)} marked as moved, "
          f"{len(all_moved_connections)} logged connections")
    
    for t in range(num_frames):
        frame = video[0, t].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 动态绘制连接线和距离文本
        connection_count = 0
        for region_idx, region in enumerate(regions):
            start_idx = region['point_offset']
            end_idx = start_idx + region['queries'].shape[0]
            unmoved_color, moved_color = region['colors']
            for p in range(start_idx, end_idx):
                local_p = p - start_idx
                if not visibility[0, t, p].item():
                    continue
                x1, y1 = tracks[0, t, p, 0].item(), tracks[0, t, p, 1].item()
                for n in region['neighbors'][local_p]:
                    global_n = n + start_idx
                    if not visibility[0, t, global_n].item() or global_n < p:
                        continue
                    x2, y2 = tracks[0, t, global_n, 0].item(), tracks[0, t, global_n, 1].item()
                    current_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    initial_dist = region['initial_distances'].get((local_p, n), current_dist)
                    relative_change = abs(current_dist - initial_dist)
                    if relative_change > threshold:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=moved_color, thickness=1)
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        text_pos = (int(mid_x + 10), int(mid_y - 10))
                        text_pos = (min(max(0, text_pos[0]), width - 80), min(max(0, text_pos[1]), height - 20))
                        cv2.putText(frame, f"{current_dist:.2f} px", text_pos, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, moved_color, 1)
                        connection_count += 1
                        if t < 5:
                            print(f"Frame {t}, Region {region_idx}: Point {local_p} to {n}: {current_dist:.2f} px at {text_pos}")
                    else:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=unmoved_color, thickness=1)
                        if t == 0:
                            print(f"Frame {t}, Region {region_idx}: {COLOR_NAMES[unmoved_color]} line from Point {local_p} to {n}, "
                                  f"relative change: {relative_change:.4f} px")
        
        # 绘制点
        for p in range(total_points):
            if visibility[0, t, p].item():
                x, y = int(tracks[0, t, p, 0].item()), int(tracks[0, t, p, 1].item())
                for region_idx, region in enumerate(regions):
                    start_idx = region['point_offset']
                    if start_idx <= p < start_idx + region['queries'].shape[0]:
                        unmoved_color, moved_color = region['colors']
                        color = moved_color if p in all_moved_points else unmoved_color
                        cv2.circle(frame, (x, y), 5, color, -1)
                        if t == 0:
                            local_p = p - start_idx
                            print(f"Region {region_idx}, Point {local_p}: "
                                  f"{'Moved (' + COLOR_NAMES[moved_color] + ')' if p in all_moved_points else 'Unmoved (' + COLOR_NAMES[unmoved_color] + ')'}, "
                                  f"Position: ({x}, {y})")
                        break
        
        print(f"Frame {t}: Drew {connection_count} connections")
        out.write(frame)
        del frame
    
    out.release()
    print(f"Custom video saved to {save_path}")

if __name__ == "__main__":
    print("\n" + '-' * 10 + " Section starts here " + '-' * 10)
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="/home/surgicalai/Data/test5.1-1.mp4", help="path to a video")
    parser.add_argument("--json_path", default="000000_tracking_points.json", help="path to JSON file with tracking points")
    parser.add_argument("--region_id", type=int, default=1, help="region ID to track points from")
    parser.add_argument("--neighbor_distance", type=float, default=20, help="Max distance for neighbor connections (px)")
    parser.add_argument("--threshold", type=float, default=20, help="Threshold for movement detection in px")
    parser.add_argument("--checkpoint", default=None, help="CoTracker model parameters")
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument("--grid_query_frame", type=int, default=0, help="Compute dense and grid tracks starting from this frame")
    parser.add_argument("--backward_tracking", action="store_true", help="Compute tracks in both directions")
    parser.add_argument("--use_v2_model", action="store_true", help="Use CoTracker2 instead of CoTracker++")
    parser.add_argument("--offline", action="store_true", help="Use the offline model")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of frames per chunk")
    parser.add_argument("--save_dir", default="/home/surgicalai/Data/output/co-tracker", help="Directory to save output videos")

    args = parser.parse_args()
    print(f"Arguments parsed: video_path={args.video_path}, json_path={args.json_path}, region_id={args.region_id}, "
          f"chunk_size={args.chunk_size}, save_dir={args.save_dir}")
    print(f"Neighbor distance={args.neighbor_distance}, Threshold={args.threshold}")

    # 验证视频文件
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")

    # 获取视频尺寸
    probe = ffmpeg.probe(args.video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    # 从 JSON 文件加载跟踪点
    points = load_points_from_json(args.json_path, args.region_id, width, height)
    if not points:
        raise ValueError(f"No valid points loaded for region {args.region_id}")

    # 初始化区域（仅一个区域，包含 JSON 指定的点）
    regions = []
    total_points = 0
    queries, neighbors, initial_distances, point_to_index = generate_tracking_points(
        points, width=width, height=height, region_idx=args.region_id, neighbor_distance=args.neighbor_distance
    )
    regions.append({
        'queries': queries,
        'neighbors': neighbors,
        'initial_distances': initial_distances,
        'point_to_index': point_to_index,
        'point_offset': total_points,
        'colors': REGION_COLORS[0]  # 使用第一组颜色
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
            
            # 更新 queries（后续 chunk 使用 last_positions）
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
                        'neighbors': region['neighbors'],
                        'initial_distances': region['initial_distances'],
                        'point_to_index': region['point_to_index'],
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
            print(f"Total tracking points: {total_points}, Regions: {len(regions)}")
            
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
            print("Running model inference with tracking points")
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

            # 为每个区域处理移动和日志，更新 last_positions
            for region_idx, region in enumerate(regions):
                start_idx = region['point_offset']
                end_idx = start_idx + region['queries'].shape[0]
                region_tracks = pred_tracks[:, :, start_idx:end_idx]
                region_visibility = pred_visibility[:, :, start_idx:end_idx]
                
                movement_matrix, moved_points, moved_connections, initial_distances, last_positions = log_movement_matrix(
                    region_tracks, region_visibility, chunk_idx, global_frame_offset, 
                    region['neighbors'], region['initial_distances'], region['point_to_index'], 
                    args.threshold, region_idx
                )
                region['moved_points'] = {p + start_idx for p in moved_points}
                region['moved_connections'] = [
                    {
                        'frame': conn['frame'],
                        'point_idx': conn['point_idx'] + start_idx,
                        'neighbor_idx': conn['neighbor_idx'] + start_idx,
                        'relative_change': conn['relative_change']
                    } for conn in moved_connections
                ]
                region['initial_distances'] = initial_distances
                region['last_positions'] = last_positions
            
            global_frame_offset += video_chunk.shape[0]

            # 自定义可视化
            print("Visualizing chunk with custom colors, dynamic lines, and distance text")
            chunk_save_path = os.path.join(save_dir, f"{seq_name}_chunk_{chunk_idx}.mp4")
            custom_visualize(video, pred_tracks, pred_visibility, regions, chunk_save_path, args.threshold)
            
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