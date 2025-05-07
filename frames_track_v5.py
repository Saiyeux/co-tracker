# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import argparse
import depthai as dai
import numpy as np
import cv2
import logging
import os
import imageio.v3 as iio
import time
import pynvml
import sys

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor

# 日志文件以时间戳命名
log_filename = f"output_{time.strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_filename, "w")
sys.stdout = log_file

# 记录程序开始时间
start_time = time.time()

# 配置日志格式，记录时间、级别和消息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 根据硬件可用性选择默认设备（优先 CUDA，其次 MPS，最后 CPU）
DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# 颜色对：(未移动色, 移动色)，BGR 格式
REGION_COLORS = [
    ((255, 0, 0), (0, 0, 255)),  # 蓝色未移动，红色移动
]

# 颜色名称映射，用于日志输出
COLOR_NAMES = {
    (255, 0, 0): 'Blue',
    (0, 0, 255): 'Red',
}

def setup_rgb_camera(pipeline, width=1280, height=720, fps=30):
    """
    配置 DepthAI RGB 相机节点。

    参数:
        pipeline: DepthAI 管道对象
        width: 相机预览宽度，默认为 1280
        height: 相机预览高度，默认为 720
        fps: 相机帧率，默认为 30

    返回:
        str: 相机输出流的名称（"rgb"）
    """
    camera = pipeline.create(dai.node.ColorCamera)
    camera.setPreviewSize(width, height)
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camera.setInterleaved(False)
    camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camera.setFps(fps)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    camera.preview.link(xout.input)

    return "rgb"

def initialize_nvml():
    """
    初始化 NVML 用于 GPU 监控。

    返回:
        handle: NVML 设备句柄，若初始化失败则返回 None
    """
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
    """
    获取 GPU 内存使用情况（已用/总计，单位 MiB）。

    参数:
        handle: NVML 设备句柄

    返回:
        tuple: (已用内存, 总内存)，若失败则返回 (None, None)
    """
    if handle is None:
        return None, None
    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = mem_info.used // 1024 // 1024
        total_mb = mem_info.total // 1024 // 1024
        return used_mb, total_mb
    except pynvml.NVMLError as e:
        print(f"Failed to get GPU memory info: {e}")
        return None, None

def generate_grid_points(quad_points, spacing, frame_number=0, width=1280, height=720):
    """
    生成四边形区域内的网格点，并计算邻居关系和初始距离。

    参数:
        quad_points: 四边形四个顶点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        spacing: 网格点间距
        frame_number: 查询帧编号
        width: 图像宽度
        height: 图像高度

    返回:
        queries: 张量，形状 (N, 3)，包含 (t, x, y) 的查询点
        grid_shape: 网格形状 (rows, cols)
        neighbors: 字典，每个点的邻居索引列表
        initial_distances: 字典，点对之间的初始距离
        point_to_index: 字典，网格坐标到点索引的映射
    """
    # 获取四边形边界
    x_coords = [p[0] for p in quad_points]
    y_coords = [p[1] for p in quad_points]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # 限制边界在图像范围内
    min_x, max_x = max(0, min_x), min(width, max_x)
    min_y, max_y = max(0, min_y), min(height, max_y)
    
    # 生成网格点坐标
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
    
    # 将查询点转换为张量
    queries = torch.tensor(queries, device=DEFAULT_DEVICE)
    if queries.shape[0] == 0:
        raise ValueError("No valid grid points within video boundaries")
    
    neighbors = {}
    initial_distances = {}
    # 将张量移到 CPU 用于 NumPy 计算
    queries_cpu = queries.cpu().numpy()
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if (i, j) not in point_to_index:
                continue
            idx = point_to_index[(i, j)]
            neighbor_list = []
            # 检查上下左右邻居
            if i > 0 and (i-1, j) in point_to_index:
                neighbor_idx = point_to_index[(i-1, j)]
                neighbor_list.append(neighbor_idx)
                dist = np.sqrt((queries_cpu[idx, 1] - queries_cpu[neighbor_idx, 1])**2 + 
                               (queries_cpu[idx, 2] - queries_cpu[neighbor_idx, 2])**2)
                initial_distances[(idx, neighbor_idx)] = dist
            if i < grid_shape[0] - 1 and (i+1, j) in point_to_index:
                neighbor_idx = point_to_index[(i+1, j)]
                neighbor_list.append(neighbor_idx)
                dist = np.sqrt((queries_cpu[idx, 1] - queries_cpu[neighbor_idx, 1])**2 + 
                               (queries_cpu[idx, 2] - queries_cpu[neighbor_idx, 2])**2)
                initial_distances[(idx, neighbor_idx)] = dist
            if j > 0 and (i, j-1) in point_to_index:
                neighbor_idx = point_to_index[(i, j-1)]
                neighbor_list.append(neighbor_idx)
                dist = np.sqrt((queries_cpu[idx, 1] - queries_cpu[neighbor_idx, 1])**2 + 
                               (queries_cpu[idx, 2] - queries_cpu[neighbor_idx, 2])**2)
                initial_distances[(idx, neighbor_idx)] = dist
            if j < grid_shape[1] - 1 and (i, j+1) in point_to_index:
                neighbor_idx = point_to_index[(i, j+1)]
                neighbor_list.append(neighbor_idx)
                dist = np.sqrt((queries_cpu[idx, 1] - queries_cpu[neighbor_idx, 1])**2 + 
                               (queries_cpu[idx, 2] - queries_cpu[neighbor_idx, 2])**2)
                initial_distances[(idx, neighbor_idx)] = dist
            neighbors[idx] = neighbor_list
    
    print(f"生成了 {len(queries)} 个网格点，间距 {spacing:.2f}，网格形状：{grid_shape}")
    return queries, grid_shape, neighbors, initial_distances, point_to_index

def generate_queries_from_last_positions(last_positions, frame_number=0, width=1280, height=720):
    """
    从上一窗口的最后位置生成查询点。

    参数:
        last_positions: 上一窗口的最后点位置，形状 (N, 2)
        frame_number: 查询帧编号
        width: 图像宽度
        height: 图像高度

    返回:
        queries: 张量，形状 (N, 3)，包含 (t, x, y) 的查询点
    """
    num_points = len(last_positions)
    queries = torch.zeros((num_points, 3), device=DEFAULT_DEVICE)
    queries[:, 0] = frame_number
    queries[:, 1:3] = torch.tensor(last_positions, device=DEFAULT_DEVICE)
    
    # 检查点坐标是否在图像范围内
    valid_mask = (0 <= queries[:, 1]) & (queries[:, 1] < width) & (0 <= queries[:, 2]) & (queries[:, 2] < height)
    if not valid_mask.all():
        invalid_count = (~valid_mask).sum().item()
        print(f"警告：{invalid_count} 个点坐标无效，超出 [0, {width}]x[0, {height}]")
    
    return queries

def log_movement_matrix(pred_tracks, pred_visibility, frame_count, grid_shape, neighbors, initial_distances, point_to_index, threshold, queries):
    """
    记录移动矩阵并更新最后位置。

    参数:
        pred_tracks: 预测的追踪点坐标，形状 (1, T, N, 2)
        pred_visibility: 点的可见性，形状 (1, T, N)
        frame_count: 当前帧计数
        grid_shape: 网格形状 (rows, cols)
        neighbors: 每个点的邻居索引列表
        initial_distances: 点对之间的初始距离
        point_to_index: 网格坐标到点索引的映射
        threshold: 移动检测阈值（像素）
        queries: 当前查询点，形状 (N, 3)，用于不可见点的回退

    返回:
        movement_matrix: 移动矩阵，标记移动点
        moved_points: 移动点的集合
        moved_connections: 超阈值的连接线信息
        initial_distances: 更新后的初始距离
        last_positions: 最后位置数组
    """
    num_frames = pred_tracks.shape[1]
    num_points = pred_tracks.shape[2]
    
    # 初始化移动矩阵和移动点集合
    movement_matrix = np.zeros(grid_shape, dtype=int)
    moved_points = set()
    moved_info = {}
    distance_changes = []
    moved_connections = []
    
    # 遍历每帧和每个网格点
    for frame_idx in range(num_frames):
        global_frame = frame_count - num_frames + frame_idx
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                if (i, j) not in point_to_index:
                    continue
                point_idx = point_to_index[(i, j)]
                if point_idx in moved_info:
                    movement_matrix[i, j] = 1
                    moved_points.add(point_idx)
                    continue
                x1, y1 = pred_tracks[0, frame_idx, point_idx, 0].item(), pred_tracks[0, frame_idx, point_idx, 1].item()
                
                # 计算与邻居的最大相对距离变化
                max_relative_change = 0.0
                max_neighbor_idx = None
                for neighbor_idx in neighbors[point_idx]:
                    x2, y2 = pred_tracks[0, frame_idx, neighbor_idx, 0].item(), pred_tracks[0, frame_idx, neighbor_idx, 1].item()
                    current_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    initial_dist = initial_distances.get((point_idx, neighbor_idx), current_dist)
                    relative_change = current_dist - initial_dist
                    
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
                
                distance_changes.append(max_relative_change)
                if max_relative_change > threshold:
                    movement_matrix[i, j] = 1
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
    
    # 记录移动点信息
    print(f"\n移动点数量：{len(moved_info)}")
    if moved_info:
        print("移动点详情：")
        for point_idx, move in moved_info.items():
            print(f"点 {point_idx}：帧 {move['frame']}：(x={move['x']:.2f}, y={move['y']:.2f}), "
                  f"与邻居 {move['neighbor_idx']} 的相对距离变化：{move['relative_change']:.4f} 像素 "
                  f"(当前：{move['current_dist']:.2f}, 初始：{move['initial_dist']:.2f})")
    
    # 记录移动连接信息
    print(f"\n移动连接数量：{len(moved_connections)}")
    if moved_connections:
        print("移动连接详情：")
        for conn in moved_connections:
            print(f"帧 {conn['frame']}：点 {conn['point_idx']} 到邻居 {conn['neighbor_idx']}, "
                  f"相对变化：{conn['relative_change']:.4f} 像素")
    
    # 记录距离变化统计
    if distance_changes:
        print(f"距离变化统计：最小={min(distance_changes):.4f}, 最大={max(distance_changes):.4f}, "
              f"平均={np.mean(distance_changes):.4f}, 标准差={np.std(distance_changes):.4f}")
    
    # 更新最后位置
    last_positions = np.zeros((num_points, 2))
    for i in range(num_points):
        if pred_visibility[0, -1, i].item():
            last_positions[i] = pred_tracks[0, -1, i, :].cpu().numpy()
        else:
            last_positions[i] = queries[i, 1:3].cpu().numpy()
    
    return movement_matrix, moved_points, moved_connections, initial_distances, last_positions

def custom_visualize(frame, tracks, visibility, moved_points, width, height, unmoved_color, moved_color):
    """
    可视化帧，仅绘制追踪点（移动点为红色，未移动点为蓝色）。

    参数:
        frame: 输入帧（RGB 格式）
        tracks: 追踪点坐标，形状 (1, T, N, 2)
        visibility: 点的可见性，形状 (1, T, N)
        moved_points: 移动点的集合
        width: 图像宽度
        height: 图像高度
        unmoved_color: 未移动点的颜色（BGR 格式）
        moved_color: 移动点的颜色（BGR 格式）

    返回:
        frame: 绘制了追踪点的帧（RGB 格式）
    """
    # 转换为 BGR 格式以进行 OpenCV 绘制
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # 绘制可见的追踪点
    for p in range(tracks.shape[2]):
        if visibility[0, 0, p].item():
            x, y = int(tracks[0, 0, p, 0].item()), int(tracks[0, 0, p, 1].item())
            color = moved_color if p in moved_points else unmoved_color
            cv2.circle(frame, (x, y), 5, color, -1)
    
    print(f"绘制了 {tracks.shape[2]} 个追踪点")
    # 转换回 RGB 格式
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def _process_step(window_frames, is_first_step, model, queries, grid_query_frame, device):
    """
    处理一组帧以获取追踪预测。

    参数:
        window_frames: 窗口帧列表
        is_first_step: 是否为首次处理
        model: CoTracker 模型
        queries: 查询点张量
        grid_query_frame: 网格查询帧编号
        device: 计算设备

    返回:
        tuple: (预测轨迹, 可见性)，若失败则返回 (None, None)
    """
    try:
        # 将帧堆叠并转换为张量
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=device
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )
        # 运行模型推理
        result = model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries[None],
            grid_query_frame=grid_query_frame,
        )
        return result
    except Exception as e:
        print(f"_process_step 错误：{e}")
        return None, None

def main():
    """
    主函数，处理相机或视频输入，执行追踪并可视化结果。

    功能:
        - 解析命令行参数
        - 初始化模型和相机/视频输入
        - 生成网格点并追踪移动
        - 可视化追踪点并记录日志
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_type",
        choices=["camera", "video"],
        default="camera",
        help="输入类型：相机或视频",
    )
    parser.add_argument(
        "--video_path",
        default="./assets/sample.mp4",
        help="视频文件路径（input_type 为 video 时需要）",
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/scaled_online.pth",
        help="CoTracker 模型参数",
    )
    parser.add_argument(
        "--quad_points",
        nargs=8,
        type=float,
        default=(600, 100, 800, 100, 800, 300, 600, 300),
        help="定义四边形的四个点 (x1 y1 x2 y2 x3 y3 x4 y4，左上、右上、右下、左下)",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=50,
        help="网格点间距",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10,
        help="移动检测阈值（像素）",
    )
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="开始追踪的帧",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="相机分辨率宽度（用于相机输入）",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="相机分辨率高度（用于相机输入）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="相机帧率（用于相机输入）",
    )

    args = parser.parse_args()
    print(f"参数解析：input_type={args.input_type}, video_path={args.video_path}, "
          f"quad_points={args.quad_points}, spacing={args.spacing}, threshold={args.threshold}")

    # 检查视频文件是否存在
    if args.input_type == "video" and not os.path.isfile(args.video_path):
        raise ValueError("视频文件不存在")

    # 构造四边形顶点
    quad_points = [
        [args.quad_points[0], args.quad_points[1]],
        [args.quad_points[2], args.quad_points[3]],
        [args.quad_points[4], args.quad_points[5]],
        [args.quad_points[6], args.quad_points[7]],
    ]
    for x, y in quad_points:
        if not (0 <= x <= args.width and 0 <= y <= args.height):
            print("四边形点必须在图像范围内")
            exit(1)

    # 初始化 CoTracker 模型
    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)
    print(f"模型初始化并移到 {DEFAULT_DEVICE}")

    # 初始化可视化器和 GPU 监控
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    gpu_handle = initialize_nvml()

    # 生成初始网格点
    queries, grid_shape, neighbors, initial_distances, point_to_index = generate_grid_points(
        quad_points, args.spacing, width=args.width, height=args.height
    )
    # 初始化区域数据
    region = {
        'queries': queries,
        'grid_shape': grid_shape,
        'neighbors': neighbors,
        'initial_distances': initial_distances,
        'point_to_index': point_to_index,
        'point_offset': 0,
        'colors': REGION_COLORS[0],
        'moved_points': set(),
        'moved_connections': [],
    }

    # 初始化帧窗口和状态变量
    window_frames = []
    is_first_step = True
    frame_count = 0
    pred_tracks = None
    pred_visibility = None
    processing_times = []
    last_positions = None
    total_model_time = 0.0

    if args.input_type == "camera":
        # 配置相机输入
        pipeline = dai.Pipeline()
        stream_name = setup_rgb_camera(pipeline, args.width, args.height, args.fps)
        with dai.Device(pipeline) as device:
            queue = device.getOutputQueue(name=stream_name, maxSize=4, blocking=False)
            while True:
                step_start_time = time.time()
                frame_data = queue.tryGet()
                if frame_data is not None:
                    # 获取并转换相机帧
                    frame_bgr = frame_data.getCvFrame()
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    window_frames.append(frame_rgb)
                    frame_count += 1

                    # 每 model.step 帧处理一次
                    if frame_count % model.step == 0 and frame_count >= model.step:
                        if len(window_frames) < model.step * 2:
                            print(f"帧不足 ({len(window_frames)}/{model.step * 2})")
                            continue
                        # 使用上一窗口的最后位置更新查询点
                        if not is_first_step and last_positions is not None:
                            queries = generate_queries_from_last_positions(
                                last_positions, width=args.width, height=args.height
                            )
                            region['queries'] = queries

                        # 记录推理前 GPU 内存
                        used_mb_before, total_mb = get_gpu_memory(gpu_handle)
                        if used_mb_before is not None:
                            print(f"推理前：GPU 内存使用：{used_mb_before:.2f} MB / {total_mb:.2f} MB")

                        # 执行模型推理
                        model_start_time = time.time()
                        pred_tracks, pred_visibility = _process_step(
                            window_frames, is_first_step, model, queries, args.grid_query_frame, DEFAULT_DEVICE
                        )
                        model_time = time.time() - model_start_time
                        total_model_time += model_time
                        is_first_step = False

                        # 记录推理后 GPU 内存
                        used_mb_after, _ = get_gpu_memory(gpu_handle)
                        if used_mb_after is not None and used_mb_before is not None:
                            print(f"推理后：GPU 内存使用：{used_mb_after:.2f} MB / {total_mb:.2f} MB, "
                                  f"变化：{used_mb_after - used_mb_before:+.2f} MB")

                        # 处理追踪结果
                        if pred_tracks is not None and pred_visibility is not None:
                            movement_matrix, moved_points, moved_connections, initial_distances, last_positions = log_movement_matrix(
                                pred_tracks, pred_visibility, frame_count, grid_shape, neighbors, 
                                initial_distances, point_to_index, args.threshold, region['queries']
                            )
                            region['moved_points'].update(moved_points)
                            region['moved_connections'].extend(moved_connections)
                            region['initial_distances'] = initial_distances
                            region['last_positions'] = last_positions
                        # 保留最近的窗口帧
                        window_frames = window_frames[-model.step * 2 :]

                    # 可视化当前帧
                    if pred_tracks is not None and pred_visibility is not None:
                        curr_tracks = pred_tracks[:, -1:, :, :]
                        curr_visibility = pred_visibility[:, -1:, :]
                        initial_tracks = pred_tracks[:, args.grid_query_frame:args.grid_query_frame+1, :, :]
                        visible_mask = curr_visibility[0, 0, :] > 0.5
                        if visible_mask.sum() > 0:
                            # 计算四边形位移
                            displacement = (
                                curr_tracks[0, 0, visible_mask, :] - 
                                initial_tracks[0, 0, visible_mask, :]
                            ).mean(dim=0)
                            curr_quad_points = [
                                [p[0] + displacement[0].item(), p[1] + displacement[1].item()]
                                for p in quad_points
                            ]
                            curr_quad_points = [
                                [max(0, min(p[0], args.width)), max(0, min(p[1], args.height))]
                                for p in curr_quad_points
                            ]
                            quad_points = curr_quad_points

                        # 绘制追踪点
                        frame_rgb_with_points = custom_visualize(
                            frame_rgb, curr_tracks, curr_visibility, region['moved_points'],
                            args.width, args.height, region['colors'][0], region['colors'][1]
                        )
                        res_frame_bgr = cv2.cvtColor(frame_rgb_with_points, cv2.COLOR_RGB2BGR)
                    else:
                        # 无追踪结果时直接显示原始帧
                        res_frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    # 显示帧
                    cv2.imshow("CoTracker", res_frame_bgr)

                    # 记录处理时间
                    processing_time = time.time() - step_start_time
                    processing_times.append(processing_time)
                    print(f"帧 {frame_count} 处理时间：{processing_time:.3f} 秒")

                    # 每 100 帧输出平均 FPS
                    if frame_count % 100 == 0 and processing_times:
                        avg_processing_time = sum(processing_times) / len(processing_times)
                        avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                        print(f"最近 {len(processing_times)} 帧平均 FPS：{avg_fps:.2f}")

                    # 按 'q' 键退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        if processing_times:
                            avg_processing_time = sum(processing_times) / len(processing_times)
                            avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                            print(f"最终平均 FPS：{avg_fps:.2f}")
                        break

            # 关闭显示窗口
            cv2.destroyAllWindows()

    else:  # input_type == "video"
        # 处理视频输入
        for frame in iio.imiter(args.video_path, plugin="FFMPEG"):
            step_start_time = time.time()
            window_frames.append(frame)
            frame_count += 1

            # 每 model.step 帧处理一次
            if frame_count % model.step == 0 and frame_count >= model.step:
                if not is_first_step and last_positions is not None:
                    queries = generate_queries_from_last_positions(
                        last_positions, width=args.width, height=args.height
                    )
                    region['queries'] = queries

                # 记录推理前 GPU 内存
                used_mb_before, total_mb = get_gpu_memory(gpu_handle)
                if used_mb_before is not None:
                    print(f"推理前：GPU 内存使用：{used_mb_before:.2f} MB / {total_mb:.2f} MB")

                # 执行模型推理
                model_start_time = time.time()
                pred_tracks, pred_visibility = _process_step(
                    window_frames, is_first_step, model, queries, args.grid_query_frame, DEFAULT_DEVICE
                )
                model_time = time.time() - model_start_time
                total_model_time += model_time
                is_first_step = False

                # 记录推理后 GPU 内存
                used_mb_after, _ = get_gpu_memory(gpu_handle)
                if used_mb_after is not None and used_mb_before is not None:
                    print(f"推理后：GPU 内存使用：{used_mb_after:.2f} MB / {total_mb:.2f} MB, "
                          f"变化：{used_mb_after - used_mb_before:+.2f} MB")

                # 处理追踪结果
                if pred_tracks is not None and pred_visibility is not None:
                    movement_matrix, moved_points, moved_connections, initial_distances, last_positions = log_movement_matrix(
                        pred_tracks, pred_visibility, frame_count, grid_shape, neighbors, 
                        initial_distances, point_to_index, args.threshold, region['queries']
                    )
                    region['moved_points'].update(moved_points)
                    region['moved_connections'].extend(moved_connections)
                    region['initial_distances'] = initial_distances
                    region['last_positions'] = last_positions

            # 记录处理时间
            processing_time = time.time() - step_start_time
            processing_times.append(processing_time)
            print(f"帧 {frame_count} 处理时间：{processing_time:.3f} 秒")

        # 处理剩余帧
        if frame_count % model.step != 0:
            if not is_first_step and last_positions is not None:
                queries = generate_queries_from_last_positions(
                    last_positions, width=args.width, height=args.height
                )
                region['queries'] = queries
            pred_tracks, pred_visibility = _process_step(
                window_frames[-(frame_count % model.step) - model.step:],
                is_first_step, model, queries, args.grid_query_frame, DEFAULT_DEVICE
            )
            if pred_tracks is not None and pred_visibility is not None:
                movement_matrix, moved_points, moved_connections, initial_distances, last_positions = log_movement_matrix(
                    pred_tracks, pred_visibility, frame_count, grid_shape, neighbors, 
                    initial_distances, point_to_index, args.threshold, region['queries']
                )
                region['moved_points'].update(moved_points)
                region['moved_connections'].extend(moved_connections)
                region['initial_distances'] = initial_distances
                region['last_positions'] = last_positions

        # 保存追踪视频
        seq_name = args.video_path.split("/")[-1]
        video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(0, 3, 1, 2)[None]
        vis.visualize(
            video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame, filename=f"tracked_{seq_name}"
        )
        print(f"追踪视频已保存至 ./saved_videos/tracked_{seq_name}")

        # 输出最终 FPS
        if processing_times:
            avg_processing_time = sum(processing_times) / len(processing_times)
            avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            print(f"最终平均 FPS：{avg_fps:.2f}")

    # 关闭 NVML
    if gpu_handle is not None:
        pynvml.nvmlShutdown()
        print("NVML 关闭完成")

    # 输出程序总运行时间
    total_time = time.time() - start_time
    print(f"程序总运行时间：{total_time:.2f} 秒")
    print(f"模型推理总时间：{total_model_time:.2f} 秒")
    log_file.close()

if __name__ == "__main__":
    main()