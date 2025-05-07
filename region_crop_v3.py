# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
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

# 创建 logs 目录并设置日志文件路径
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/output_{time.strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_filename, "w")
sys.stdout = log_file

# 记录程序开始时间
start_time = time.time()

# 设置默认设备：优先 CUDA，其次 MPS，否则 CPU
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
    (255, 0, 0): '蓝色',
    (0, 0, 255): '红色',
    (0, 255, 0): '绿色',
    (0, 255, 255): '黄色',
    (255, 255, 0): '青色',
    (255, 0, 255): '紫色',
    (128, 128, 128): '灰色',
    (255, 128, 0): '橙色'
}

# 初始化 NVML 用于 GPU 监控
def initialize_nvml():
    try:
        pynvml.nvmlInit()
        if torch.cuda.is_available():
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return handle
        print("无可用 CUDA GPU 或 NVML 初始化失败，跳过 GPU 监控")
        return None
    except pynvml.NVMLError as e:
        print(f"NVML 初始化失败: {e}，跳过 GPU 监控")
        return None

# 获取 GPU 内存使用情况
def get_gpu_memory(handle):
    if handle is None:
        return None, None
    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = mem_info.used / 1024 / 1024
        total_mb = mem_info.total / 1024 / 1024
        return used_mb, total_mb
    except pynvml.NVMLError as e:
        print(f"获取 GPU 内存信息失败: {e}")
        return None, None

# 从 JSON 文件加载跟踪点，支持裁剪区域
def load_points_from_json(json_path, region_id, width, height, crop_region=None):
    """
    从 JSON 文件加载指定 region_id 的跟踪点，并根据裁剪区域筛选点。

    参数:
        json_path: JSON 文件路径
        region_id: 要加载的区域 ID
        width, height: 视频尺寸，用于验证
        crop_region: 元组 (x1, y1, x2, y2)，定义裁剪区域，或 None

    返回:
        points: 展平的点坐标列表 [x1, y1, x2, y2, ...]
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'regions' not in data:
            raise ValueError("JSON 文件必须包含 'regions' 键")
        
        for region in data['regions']:
            if region.get('region_id') == region_id:
                points = region.get('tracking_points', [])
                if not points:
                    raise ValueError(f"region_id {region_id} 没有跟踪点")
                
                # 展平点坐标并筛选
                flat_points = []
                for x, y in points:
                    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                        raise ValueError(f"无效的点坐标: ({x}, {y})")
                    # 检查点是否在裁剪区域内（如果提供了 crop_region）
                    if crop_region is not None:
                        x1, y1, x2, y2 = crop_region
                        if not (x1 <= x <= x2 and y1 <= y <= y2):
                            print(f"点 ({x}, {y}) 被跳过，位于裁剪区域 [{x1}, {y1}, {x2}, {y2}] 外")
                            continue
                    flat_points.extend([float(x), float(y)])
                
                if not flat_points:
                    raise ValueError(f"region_id {region_id} 在裁剪区域内没有有效点")
                
                print(f"从 region {region_id} 加载 {len(flat_points) // 2} 个点")
                return flat_points
        
        raise ValueError(f"未在 JSON 文件中找到 region_id {region_id}")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON 文件未找到: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"JSON 文件格式无效: {json_path}")
    except Exception as e:
        raise ValueError(f"加载 JSON 文件出错: {e}")

# 分块读取视频
def read_video_with_ffmpeg_chunked(video_path, chunk_size=100):
    print(f"开始读取视频: {video_path}")
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        total_frames = int(video_info['nb_frames'])
        print(f"视频信息: {width}x{height}, 总帧数: {total_frames}")
        
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
                        print(f"生成最终块，包含 {chunk.shape[0]} 帧")
                        yield chunk
                    break
                frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))
                frames.append(frame)
                frame_count += 1
                
                if len(frames) >= chunk_size:
                    chunk = np.stack(frames, axis=0)
                    print(f"生成块，包含 {chunk.shape[0]} 帧，从第 {frame_count - chunk_size} 帧开始")
                    yield chunk
                    frames = []
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        finally:
            process.stdout.close()
            process.wait()
            print("视频读取完成")
    
    except ffmpeg.Error as e:
        print(f"FFmpeg 错误: {e.stderr.decode()}")
        raise
    except Exception as e:
        print(f"读取视频出错: {e}")
        raise

# 生成跟踪点和邻居关系
def generate_tracking_points(points, frame_number=0, width=1920, height=1080, region_idx=0, neighbor_distance=20):
    """
    根据输入的 (x, y) 坐标生成跟踪点，并基于距离阈值确定邻居关系。

    参数:
        points: 坐标列表 [x1, y1, x2, y2, ...]
        frame_number: 起始帧号
        width, height: 视频尺寸
        region_idx: 区域索引，用于日志
        neighbor_distance: 邻居点最大距离

    返回:
        queries: 张量，形状 (N, 3)，包含 [frame_number, x, y]
        neighbors: 字典，映射点索引到邻居索引列表
        initial_distances: 字典，记录点对初始距离
        point_to_index: 字典，映射点索引到其位置
    """
    if len(points) % 2 != 0:
        raise ValueError("点必须成对提供 (x, y) 坐标")
    
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
            print(f"警告: 点 {i // 2} 位于 ({x}, {y})，超出视频边界 [0, {width}]x[0, {height}]")
    
    if not queries:
        raise ValueError(f"region {region_idx} 内无有效点")
    
    queries = torch.tensor(queries, dtype=torch.float32)
    print(f"生成 {len(queries)} 个跟踪点，区域 {region_idx}")
    
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
    
    print(f"计算 {len(queries)} 个点的邻居关系，最大邻居距离: {neighbor_distance}px")
    return queries, neighbors, initial_distances, point_to_index

# 从上一块的结束位置生成新查询点
def generate_queries_from_last_positions(region, region_idx, frame_number=0, width=1920, height=1080):
    if 'last_positions' not in region:
        raise ValueError(f"region {region_idx} 无 last_positions，确保已处理前一块")
    
    last_positions = region['last_positions']
    num_points = len(last_positions)
    queries = np.zeros((num_points, 3))
    queries[:, 0] = frame_number
    queries[:, 1:3] = last_positions
    
    valid_mask = (0 <= queries[:, 1]) & (queries[:, 1] < width) & (0 <= queries[:, 2]) & (queries[:, 2] < height)
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        print(f"警告: region {region_idx} 中 {invalid_count} 个点坐标无效，超出 [0, {width}]x[0, {height}]")
        queries[~valid_mask, 1:3] = region['queries'][~valid_mask, 1:3].cpu().numpy()
    
    queries = torch.tensor(queries, dtype=torch.float32)
    
    min_x, max_x = queries[:, 1].min().item(), queries[:, 1].max().item()
    min_y, max_y = queries[:, 2].min().item(), queries[:, 2].max().item()
    print(f"块 {chunk_idx}, 区域 {region_idx}: 新查询点范围 x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")
    
    return queries

# 计算间距变化，记录移动点，更新 last_positions
def log_movement_matrix(pred_tracks, pred_visibility, chunk_idx, global_frame_offset, neighbors, initial_distances, point_to_index, threshold, region_idx=0):
    print(f"\n处理块 {chunk_idx}, 区域 {region_idx} 的移动")
    num_frames = pred_tracks.shape[1]
    num_points = pred_tracks.shape[2]
    
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
    
    print(f"\n区域 {region_idx} 移动点数量: {len(moved_info)}")
    if moved_info:
        print(f"区域 {region_idx} 移动点详情:")
        for point_idx, move in moved_info.items():
            print(f"点 {point_idx}: 帧 {move['frame']}: (x={move['x']:.2f}, y={move['y']:.2f}), "
                  f"与邻居 {move['neighbor_idx']} 的相对距离变化: {move['relative_change']:.4f} px "
                  f"(当前: {move['current_dist']:.2f}, 初始: {move['initial_dist']:.2f})")
    else:
        print(f"警告: 区域 {region_idx} 未检测到移动点，考虑降低阈值 (当前 {threshold} px)")
    
    print(f"\n区域 {region_idx} 移动连接数量: {len(moved_connections)}")
    if moved_connections:
        print(f"区域 {region_idx} 移动连接详情:")
        for conn in moved_connections:
            print(f"帧 {conn['frame']}: 点 {conn['point_idx']} 到邻居 {conn['neighbor_idx']}, "
                  f"相对变化: {conn['relative_change']:.4f} px")
    
    if distance_changes:
        print(f"区域 {region_idx} 距离变化统计: 最小={min(distance_changes):.4f}, 最大={max(distance_changes):.4f}, "
              f"均值={np.mean(distance_changes):.4f}, 标准差={np.std(distance_changes):.4f}")
    
    last_positions = np.zeros((num_points, 2))
    for i in range(num_points):
        if pred_visibility[0, -1, i].item():
            last_positions[i] = pred_tracks[0, -1, i, :].cpu().numpy()
        else:
            last_positions[i] = (region['last_positions'][i] if 'last_positions' in region 
                                 else region['queries'][i, 1:3].cpu().numpy())
    print(f"块 {chunk_idx}, 区域 {region_idx}: 更新 {len(last_positions)} 个点的最后位置")
    
    return movement_matrix, moved_points, moved_connections, initial_distances, last_positions

# 自定义可视化，显示动态裁剪区域框和点坐标
def custom_visualize(video, tracks, visibility, regions, save_path, threshold=0.5, fps=30, crop_region=None):
    """
    生成可视化视频，绘制跟踪点、连接线、动态裁剪区域框和点坐标。

    参数:
        video: 视频张量
        tracks: 跟踪轨迹
        visibility: 点可见性
        regions: 区域信息列表
        save_path: 输出视频路径
        threshold: 移动检测阈值
        fps: 输出视频帧率
        crop_region: 初始裁剪区域 (x1, y1, x2, y2)，用于日志，或 None
    """
    print(f"生成自定义视频: {save_path}")
    height, width = video.shape[3], video.shape[4]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    num_frames = min(video.shape[1], tracks.shape[1])
    total_points = tracks.shape[2]
    
    all_moved_points = set()
    all_moved_connections = []
    for region in regions:
        all_moved_points.update(region['moved_points'])
        all_moved_connections.extend(region['moved_connections'])
    
    print(f"可视化 {total_points} 个点，跨 {len(regions)} 个区域，标记为移动的点: {len(all_moved_points)}，记录的连接: {len(all_moved_connections)}")
    
    for t in range(num_frames):
        frame = video[0, t].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 计算动态裁剪区域框
        visible_points = []
        for p in range(total_points):
            if visibility[0, t, p].item():
                x, y = tracks[0, t, p, 0].item(), tracks[0, t, p, 1].item()
                visible_points.append((x, y))
        
        if visible_points:
            # 计算所有可见点的最小外接矩形
            x_coords, y_coords = zip(*visible_points)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            # 扩展边界 20 像素
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)
            # 绘制动态框
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 255, 255), 2)  # 蓝色框，粗细 2
            print(f"帧 {t}: 绘制动态裁剪区域框 [{x_min:.0f}, {y_min:.0f}, {x_max:.0f}, {y_max:.0f}]")
        else:
            print(f"帧 {t}: 无可见点，跳过动态框绘制")
        
        # 记录初始裁剪区域（仅第一帧）
        if t == 0 and crop_region is not None:
            x1, y1, x2, y2 = [int(v) for v in crop_region]
            print(f"帧 {t}: 初始裁剪区域 [{x1}, {y1}, {x2}, {y2}]（仅用于点筛选）")
        
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
                        # cv2.putText(frame, f"{current_dist:.2f} px", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, moved_color, 1)
                        connection_count += 1
                        if t < 5:
                            print(f"帧 {t}, 区域 {region_idx}: 点 {local_p} 到 {n}: {current_dist:.2f} px，位置 {text_pos}")
                    else:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=unmoved_color, thickness=1)
                        if t == 0:
                            print(f"帧 {t}, 区域 {region_idx}: {COLOR_NAMES[unmoved_color]} 线，点 {local_p} 到 {n}, "
                                  f"相对变化: {relative_change:.4f} px")
        
        # 绘制点和坐标
        for p in range(total_points):
            if visibility[0, t, p].item():
                x, y = int(tracks[0, t, p, 0].item()), int(tracks[0, t, p, 1].item())
                for region_idx, region in enumerate(regions):
                    start_idx = region['point_offset']
                    if start_idx <= p < start_idx + region['queries'].shape[0]:
                        unmoved_color, moved_color = region['colors']
                        color = moved_color if p in all_moved_points else unmoved_color
                        cv2.circle(frame, (x, y), 5, color, -1)
                        # 绘制点坐标
                        text_pos = (x + 10, y - 10)
                        text_pos = (min(max(0, text_pos[0]), width - 80), min(max(0, text_pos[1]), height - 20))
                        # cv2.putText(frame, f"({x:.0f}, {y:.0f})", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        if t == 0:
                            local_p = p - start_idx
                            print(f"区域 {region_idx}, 点 {local_p}: "
                                  f"{'移动 (' + COLOR_NAMES[moved_color] + ')' if p in all_moved_points else '未移动 (' + COLOR_NAMES[unmoved_color] + ')'}, "
                                  f"位置: ({x}, {y}), 坐标标注位置: {text_pos}")
                        break
        
        print(f"帧 {t}: 绘制 {connection_count} 条连接")
        out.write(frame)
        del frame
    
    out.release()
    print(f"自定义视频保存至 {save_path}")

if __name__ == "__main__":
    print("\n" + '-' * 10 + " 程序开始 " + '-' * 10)
    parser = argparse.ArgumentParser(description="视频点跟踪，支持动态裁剪区域和可视化")
    parser.add_argument("--video_path", default="/home/surgicalai/Data/test5.1-1.mp4", help="视频文件路径")
    parser.add_argument("--json_path", default="000000_tracking_points.json", help="跟踪点 JSON 文件路径")
    parser.add_argument("--region_id", type=int, default=1, help="跟踪点的区域 ID")
    parser.add_argument("--neighbor_distance", type=float, default=20, help="邻居连接的最大距离 (px)")
    parser.add_argument("--threshold", type=float, default=20, help="移动检测的阈值 (px)")
    parser.add_argument("--checkpoint", default=None, help="CoTracker 模型参数")
    parser.add_argument("--grid_size", type=int, default=10, help="规则网格大小")
    parser.add_argument("--grid_query_frame", type=int, default=0, help="从此帧开始计算密集和网格跟踪")
    parser.add_argument("--backward_tracking", action="store_true", help="双向计算跟踪")
    parser.add_argument("--use_v2_model", action="store_true", help="使用 CoTracker2 而非 CoTracker++")
    parser.add_argument("--offline", action="store_true", help="使用离线模型")
    parser.add_argument("--chunk_size", type=int, default=100, help="每块帧数")
    parser.add_argument("--save_dir", default="/home/surgicalai/Data/output/co-tracker", help="输出视频保存目录")
    parser.add_argument("--crop", type=float, nargs=4, metavar=('x1', 'y1', 'x2', 'y2'),
                        help="初始裁剪区域坐标 x1,y1,x2,y2 (左上角和右下角)，用于点筛选")

    args = parser.parse_args()
    print(f"参数解析: 视频路径={args.video_path}, JSON 路径={args.json_path}, 区域 ID={args.region_id}, "
          f"块大小={args.chunk_size}, 输出目录={args.save_dir}")
    print(f"邻居距离={args.neighbor_distance}, 阈值={args.threshold}")
    if args.crop:
        print(f"初始裁剪区域: x1={args.crop[0]}, y1={args.crop[1]}, x2={args.crop[2]}, y2={args.crop[3]}")

    # 验证视频文件
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"视频文件未找到: {args.video_path}")

    # 获取视频尺寸
    probe = ffmpeg.probe(args.video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    # 验证初始裁剪区域（如果提供）
    crop_region = None
    if args.crop:
        x1, y1, x2, y2 = args.crop
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"初始裁剪区域 [{x1}, {y1}, {x2}, {y2}] 无效，视频尺寸为 {width}x{height}")
        crop_region = (x1, y1, x2, y2)

    # 从 JSON 文件加载跟踪点
    points = load_points_from_json(args.json_path, args.region_id, width, height, crop_region)
    if not points:
        raise ValueError(f"region {args.region_id} 无有效点")

    # 初始化区域（仅一个区域，包含 JSON 指定的点）
    print('Initializing region with loaded points')
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
        'colors': REGION_COLORS[0]
    })
    total_points += queries.shape[0]
    
    # 初始化 NVML
    gpu_handle = initialize_nvml()

    # 初始化模型
    print("初始化 CoTracker 模型")
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
        print(f"模型初始化完成，移动至 {DEFAULT_DEVICE}")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        raise

    # 分块处理视频
    seq_name = args.video_path.split("/")[-1].split(".")[0]
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"输出目录准备完成: {save_dir}")
    
    chunk_idx = 0
    total_model_time = 0.0
    global_frame_offset = 0

    try:
        for video_chunk in read_video_with_ffmpeg_chunked(args.video_path, chunk_size=args.chunk_size):
            pair_start_time = time.time()
            print(f"\n处理块 {chunk_idx}, 帧数: {video_chunk.shape[0]}")
            
            # 更新查询点（后续块使用 last_positions）
            if chunk_idx > 0:
                print(f"使用最后位置更新块 {chunk_idx} 的查询点")
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
            print(f"查询点定义，形状: {all_queries.shape}, 设备: {all_queries.device}")
            print(f"总跟踪点数: {total_points}, 区域数: {len(regions)}")
            
            # 转换视频块为张量
            print("将视频块转换为 PyTorch 张量")
            video = torch.from_numpy(video_chunk).permute(0, 3, 1, 2)[None].float().to(DEFAULT_DEVICE)
            print(f"块张量创建，形状: {video.shape}, 设备: {video.device}")

            # 查询 GPU 内存占用
            used_mb_before, total_mb = get_gpu_memory(gpu_handle)
            if used_mb_before is not None:
                print(
                    f"推理前 (块 {chunk_idx}): "
                    f"GPU 内存使用: {used_mb_before:.2f} MB / {total_mb:.2f} MB "
                    f"({used_mb_before / total_mb * 100:.2f}%)"
                )

            # 模型推理
            print("使用跟踪点运行模型推理")
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

            # 查询推理后 GPU 内存占用
            used_mb_after, _ = get_gpu_memory(gpu_handle)
            if used_mb_after is not None and used_mb_before is not None:
                memory_diff = used_mb_after - used_mb_before
                print(
                    f"推理后 (块 {chunk_idx}): "
                    f"GPU 内存使用: {used_mb_after:.2f} MB / {total_mb:.2f} MB "
                    f"({used_mb_after / total_mb * 100:.2f}%), "
                    f"变化: {'+' if memory_diff >= 0 else ''}{memory_diff:.2f} MB"
                )
            print(f"块 {chunk_idx} 推理完成，轨迹形状: {pred_tracks.shape}, 模型耗时: {model_time:.2f} 秒")

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

            # 自定义可视化，传递初始裁剪区域
            print("使用自定义颜色、动态线条、距离文本、动态裁剪区域框和点坐标可视化块")
            chunk_save_path = os.path.join(save_dir, f"{seq_name}_chunk_{chunk_idx}.mp4")
            custom_visualize(video, pred_tracks, pred_visibility, regions, chunk_save_path, args.threshold, crop_region=crop_region)
            
            pair_time = time.time() - pair_start_time
            print(f"(块总耗时: {pair_time:.2f} 秒)")
            
            # 清理内存
            print("清理内存")
            del video, pred_tracks, pred_visibility, video_chunk
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            chunk_idx += 1

    except Exception as e:
        print(f"处理失败: {e}")
        raise
    finally:
        # 关闭 NVML
        if gpu_handle is not None:
            pynvml.nvmlShutdown()
            print("NVML 关闭完成")

    # 计算并打印总运行时间和模型处理时间
    total_time = time.time() - start_time
    print(f"\n处理完成！")
    print(f"程序总运行时间: {total_time:.2f} 秒")
    print(f"模型推理总时间: {total_model_time:.2f} 秒")
    print('-' * 10 + " 程序结束 " + '-' * 12 + "\n")

log_file.close()