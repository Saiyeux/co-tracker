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
import json

# 创建 logs 目录并设置日志文件路径
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/output_{time.strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_filename, "w")
sys.stdout = log_file

# 记录程序开始时间
start_time = time.time()

# 设置默认设备
DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# 区域颜色对：(未移动色, 移动色)，BGR 格式
REGION_COLORS = [
    ((255, 0, 0), (0, 0, 255)),  # 蓝色未移动，红色移动
]

# 颜色名称映射
COLOR_NAMES = {
    (255, 0, 0): '蓝色',
    (0, 0, 255): '红色',
}

# 初始化 NVML
def initialize_nvml():
    try:
        pynvml.nvmlInit()
        if torch.cuda.is_available():
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return handle
        print("无可用 CUDA GPU，跳过 GPU 监控")
        return None
    except pynvml.NVMLError as e:
        print(f"NVML 初始化失败: {e}")
        return None

# 获取 GPU 内存
def get_gpu_memory(handle):
    if handle is None:
        return None, None
    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / 1024 / 1024, mem_info.total / 1024 / 1024
    except pynvml.NVMLError:
        return None, None

# 验证四边形有效性
def validate_quadrilateral(points, width, height):
    """
    验证四边形坐标是否有效（非共线、在视频范围内）。

    参数:
        points: 8个值 [x1, y1, x2, y2, x3, y3, x4, y4]
        width, height: 视频尺寸

    返回:
        quad: 2D NumPy 数组 [[x1, y1], [x2, y2], ...]
    """
    quad = np.array(points).reshape(4, 2).astype(np.float32)
    # 检查坐标范围
    if not (np.all(quad >= 0) and np.all(quad[:, 0] <= width) and np.all(quad[:, 1] <= height)):
        raise ValueError(f"四边形坐标 {points} 超出视频范围 [0, {width}]x[0, {height}]")
    # 检查面积（避免共线）
    area = cv2.contourArea(quad)
    if area < 10:
        raise ValueError(f"四边形面积过小 ({area:.2f})，可能共线")
    return quad

# 检查点是否在四边形内
def point_in_quadrilateral(point, quad):
    """
    使用 cv2.pointPolygonTest 检查点是否在四边形内。

    参数:
        point: (x, y) 坐标
        quad: 2D NumPy 数组 [[x1, y1], [x2, y2], ...]

    返回:
        bool: 点是否在四边形内
    """
    return cv2.pointPolygonTest(quad, point, False) >= 0

# 从 JSON 文件加载跟踪点
def load_points_from_json(json_path, region_id, quad=None):
    """
    加载指定 region_id 的跟踪点，筛选在四边形区域内的点。

    参数:
        json_path: JSON 文件路径
        region_id: 区域 ID
        quad: 四边形顶点数组，或 None

    返回:
        points: 展平的点坐标 [x1, y1, x2, y2, ...]
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'regions' not in data:
            raise ValueError("JSON 文件缺少 'regions' 键")
        
        for region in data['regions']:
            if region.get('region_id') == region_id:
                points = region.get('tracking_points', [])
                if not points:
                    raise ValueError(f"region_id {region_id} 无跟踪点")
                
                flat_points = []
                for x, y in points:
                    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                        raise ValueError(f"无效坐标: ({x}, {y})")
                    if quad is not None and not point_in_quadrilateral((x, y), quad):
                        print(f"点 ({x}, {y}) 在四边形外，跳过")
                        continue
                    flat_points.extend([float(x), float(y)])
                
                if not flat_points:
                    raise ValueError(f"region_id {region_id} 在四边形内无有效点")
                
                print(f"加载 region {region_id} 的 {len(flat_points) // 2} 个点")
                return flat_points
        
        raise ValueError(f"未找到 region_id {region_id}")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON 文件未找到: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"JSON 文件格式无效: {json_path}")

# 分块读取视频
def read_video_with_ffmpeg_chunked(video_path, chunk_size=100):
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width, height = int(video_info['width']), int(video_info['height'])
        total_frames = int(video_info['nb_frames'])
        print(f"视频: {width}x{height}, {total_frames} 帧")
        
        process = ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='rgb24').run_async(pipe_stdout=True)
        
        frames = []
        frame_count = 0
        while True:
            in_bytes = process.stdout.read(width * height * 3)
            if not in_bytes:
                if frames:
                    yield np.stack(frames, axis=0)
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))
            frames.append(frame)
            frame_count += 1
            if len(frames) >= chunk_size:
                yield np.stack(frames, axis=0)
                frames = []
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        process.stdout.close()
        process.wait()
        print("视频读取完成")
    
    except ffmpeg.Error as e:
        print(f"FFmpeg 错误: {e.stderr.decode()}")
        raise
    except Exception as e:
        print(f"读取视频失败: {e}")
        raise

# 生成跟踪点和邻居关系
def generate_tracking_points(points, frame_number=0, width=1920, height=1080, region_idx=0, neighbor_distance=20):
    if len(points) % 2 != 0:
        raise ValueError("点坐标必须成对")
    
    num_points = len(points) // 2
    queries = []
    point_to_index = {}
    
    for i in range(0, len(points), 2):
        x, y = points[i], points[i + 1]
        if 0 <= x < width and 0 <= y < height:
            queries.append([float(frame_number), float(x), float(y)])
            point_to_index[i // 2] = i // 2
        else:
            print(f"点 {i // 2} ({x}, {y}) 超出边界，跳过")
    
    if not queries:
        raise ValueError(f"region {region_idx} 无有效点")
    
    queries = torch.tensor(queries, dtype=torch.float32)
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
    
    print(f"生成 {len(queries)} 个跟踪点，邻居距离: {neighbor_distance}px")
    return queries, neighbors, initial_distances, point_to_index

# 从上一块生成查询点
def generate_queries_from_last_positions(region, region_idx, frame_number=0, width=1920, height=1080):
    if 'last_positions' not in region:
        raise ValueError(f"region {region_idx} 无 last_positions")
    
    last_positions = region['last_positions']
    queries = np.zeros((len(last_positions), 3))
    queries[:, 0] = frame_number
    queries[:, 1:3] = last_positions
    
    valid_mask = (0 <= queries[:, 1]) & (queries[:, 1] < width) & (0 <= queries[:, 2]) & (queries[:, 2] < height)
    if not np.all(valid_mask):
        queries[~valid_mask, 1:3] = region['queries'][~valid_mask, 1:3].cpu().numpy()
    
    return torch.tensor(queries, dtype=torch.float32)

# 处理移动点和间距变化
def log_movement_matrix(pred_tracks, pred_visibility, chunk_idx, global_frame_offset, neighbors, initial_distances, point_to_index, threshold, region_idx=0):
    num_frames, num_points = pred_tracks.shape[1], pred_tracks.shape[2]
    movement_matrix = np.zeros(num_points, dtype=int)
    moved_points = set()
    moved_connections = []
    
    for frame_idx in range(num_frames):
        global_frame = global_frame_offset + frame_idx
        for point_idx in range(num_points):
            if point_idx in moved_points:
                movement_matrix[point_idx] = 1
                continue
            x1, y1 = pred_tracks[0, frame_idx, point_idx, 0].item(), pred_tracks[0, frame_idx, point_idx, 1].item()
            for neighbor_idx in neighbors[point_idx]:
                x2, y2 = pred_tracks[0, frame_idx, neighbor_idx, 0].item(), pred_tracks[0, frame_idx, neighbor_idx, 1].item()
                current_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                initial_dist = initial_distances.get((point_idx, neighbor_idx), current_dist)
                relative_change = abs(current_dist - initial_dist)
                if relative_change > threshold:
                    movement_matrix[point_idx] = 1
                    moved_points.add(point_idx)
                    moved_connections.append({
                        'frame': global_frame,
                        'point_idx': point_idx,
                        'neighbor_idx': neighbor_idx,
                        'relative_change': relative_change
                    })
    
    last_positions = np.zeros((num_points, 2))
    for i in range(num_points):
        last_positions[i] = (pred_tracks[0, -1, i, :].cpu().numpy() if pred_visibility[0, -1, i].item()
                             else region['last_positions'][i] if 'last_positions' in region
                             else region['queries'][i, 1:3].cpu().numpy())
    
    print(f"块 {chunk_idx}, 区域 {region_idx}: {len(moved_points)} 个移动点，{len(moved_connections)} 条移动连接")
    return movement_matrix, moved_points, moved_connections, initial_distances, last_positions

# 自定义可视化
def custom_visualize(video, tracks, visibility, regions, save_path, threshold=0.5, fps=30, initial_quad=None):
    """
    生成可视化视频，绘制跟踪点、连接线、动态四边形和点坐标，区域外为黑色。

    参数:
        video: 视频张量
        tracks: 跟踪轨迹
        visibility: 点可见性
        regions: 区域信息
        save_path: 输出路径
        threshold: 移动阈值
        fps: 帧率
        initial_quad: 初始四边形顶点，或 None
    """
    print(f"生成视频: {save_path}")
    height, width = video.shape[3], video.shape[4]
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    num_frames, total_points = min(video.shape[1], tracks.shape[1]), tracks.shape[2]
    all_moved_points = set()
    for region in regions:
        all_moved_points.update(region['moved_points'])
    
    for t in range(num_frames):
        frame = video[0, t].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 创建掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        visible_points = [(tracks[0, t, p, 0].item(), tracks[0, t, p, 1].item()) 
                          for p in range(total_points) if visibility[0, t, p].item()]
        
        # 计算动态四边形
        if visible_points:
            points_array = np.array(visible_points, dtype=np.float32)
            # 使用最小外接矩形，转换为四边形
            rect = cv2.minAreaRect(points_array)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            # 扩展边界
            padding = 20
            box = box + np.array([[-padding, -padding], [padding, -padding], [padding, padding], [-padding, padding]])
            box[:, 0] = np.clip(box[:, 0], 0, width)
            box[:, 1] = np.clip(box[:, 1], 0, height)
            cv2.fillPoly(mask, [box], 255)
            print(f"帧 {t}: 动态四边形 {box.tolist()}")
        else:
            print(f"帧 {t}: 无可见点，跳过四边形")
            mask[:] = 0
        
        # 应用掩码
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # 绘制四边形边界
        if visible_points:
            cv2.polylines(frame, [box], True, (255, 255, 255), 2)  # 蓝色边界
        
        # 绘制连接线
        connection_count = 0
        for region_idx, region in enumerate(regions):
            start_idx, end_idx = region['point_offset'], region['point_offset'] + region['queries'].shape[0]
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
                    color = moved_color if relative_change > threshold else unmoved_color
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                    if relative_change > threshold:
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        text_pos = (int(mid_x + 10), int(mid_y - 10))
                        text_pos = (min(max(0, text_pos[0]), width - 80), min(max(0, text_pos[1]), height - 20))
                        cv2.putText(frame, f"{current_dist:.2f}px", text_pos, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, moved_color, 1)
                        connection_count += 1
        
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
                        text_pos = (x + 10, y - 10)
                        text_pos = (min(max(0, text_pos[0]), width - 80), min(max(0, text_pos[1]), height - 20))
                        cv2.putText(frame, f"({x:.0f}, {y:.0f})", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        break
        
        print(f"帧 {t}: {connection_count} 条连接")
        out.write(frame)
        del frame, mask
    
    out.release()
    print(f"视频保存至 {save_path}")

if __name__ == "__main__":
    print("\n" + '-' * 10 + " 程序开始 " + '-' * 10)
    parser = argparse.ArgumentParser(description="视频点跟踪，支持四边形裁剪")
    parser.add_argument("--video_path", default="/home/surgicalai/Data/test5.1-1.mp4", help="视频路径")
    parser.add_argument("--json_path", default="000000_tracking_points.json", help="跟踪点 JSON 路径")
    parser.add_argument("--region_id", type=int, default=1, help="区域 ID")
    parser.add_argument("--neighbor_distance", type=float, default=20, help="邻居距离 (px)")
    parser.add_argument("--threshold", type=float, default=20, help="移动阈值 (px)")
    parser.add_argument("--checkpoint", default=None, help="CoTracker 模型参数")
    parser.add_argument("--grid_size", type=int, default=10, help="网格大小")
    parser.add_argument("--grid_query_frame", type=int, default=0, help="网格跟踪起始帧")
    parser.add_argument("--backward_tracking", action="store_true", help="双向跟踪")
    parser.add_argument("--use_v2_model", action="store_true", help="使用 CoTracker2")
    parser.add_argument("--offline", action="store_true", help="离线模型")
    parser.add_argument("--chunk_size", type=int, default=100, help="块帧数")
    parser.add_argument("--save_dir", default="/home/surgicalai/Data/output/co-tracker", help="输出目录")
    parser.add_argument("--crop", type=float, nargs=8, metavar=('x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'),
                        help="四边形裁剪区域坐标 x1,y1,x2,y2,x3,y3,x4,y4")

    args = parser.parse_args()
    print(f"参数: 视频={args.video_path}, JSON={args.json_path}, 区域 ID={args.region_id}, 块大小={args.chunk_size}")
    if args.crop:
        print(f"初始四边形: {args.crop}")

    # 验证视频文件
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"视频未找到: {args.video_path}")

    # 获取视频尺寸
    probe = ffmpeg.probe(args.video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width, height = int(video_info['width']), int(video_info['height'])

    # 验证四边形
    initial_quad = None
    if args.crop:
        initial_quad = validate_quadrilateral(args.crop, width, height)

    # 加载跟踪点
    points = load_points_from_json(args.json_path, args.region_id, initial_quad)
    if not points:
        raise ValueError(f"region {args.region_id} 无有效点")

    # 初始化区域
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
    print("初始化 CoTracker")
    try:
        if args.checkpoint:
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=args.use_v2_model,
                offline=args.offline,
                window_len=60 if args.offline else 16,
            )
        else:
            model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        model = model.to(DEFAULT_DEVICE)
        print(f"模型加载至 {DEFAULT_DEVICE}")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        raise

    # 分块处理视频
    seq_name = args.video_path.split("/")[-1].split(".")[0]
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    chunk_idx = 0
    total_model_time = 0.0
    global_frame_offset = 0

    try:
        for video_chunk in read_video_with_ffmpeg_chunked(args.video_path, chunk_size=args.chunk_size):
            pair_start_time = time.time()
            print(f"\n块 {chunk_idx}: {video_chunk.shape[0]} 帧")
            
            # 更新查询点
            if chunk_idx > 0:
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
            
            # 合并查询点
            all_queries = torch.cat([r['queries'] for r in regions], dim=0)
            if torch.cuda.is_available():
                all_queries = all_queries.cuda()
            
            # 转换视频块
            video = torch.from_numpy(video_chunk).permute(0, 3, 1, 2)[None].float().to(DEFAULT_DEVICE)

            # 模型推理
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
            print(f"块 {chunk_idx} 推理完成，耗时: {model_time:.2f}秒")

            # 处理移动点
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
                region['moved_connections'] = moved_connections
                region['initial_distances'] = initial_distances
                region['last_positions'] = last_positions
            
            global_frame_offset += video_chunk.shape[0]

            # 可视化
            chunk_save_path = os.path.join(save_dir, f"{seq_name}_chunk_{chunk_idx}.mp4")
            custom_visualize(video, pred_tracks, pred_visibility, regions, chunk_save_path, args.threshold, initial_quad=initial_quad)
            
            print(f"块 {chunk_idx} 耗时: {time.time() - pair_start_time:.2f}秒")
            del video, pred_tracks, pred_visibility, video_chunk
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            chunk_idx += 1

    except Exception as e:
        print(f"处理失败: {e}")
        raise
    finally:
        if gpu_handle is not None:
            pynvml.nvmlShutdown()
            print("NVML 关闭")

    total_time = time.time() - start_time
    print(f"\n处理完成！总耗时: {total_time:.2f}秒，模型耗时: {total_model_time:.2f}秒")
    print('-' * 10 + " 程序结束 " + '-' * 10 + "\n")

log_file.close()