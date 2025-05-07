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
import time
import pynvml

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor

# 配置日志，设置日志级别为INFO，格式包括时间、级别和消息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置默认设备：优先使用CUDA，其次MPS，最后CPU
DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

def setup_rgb_camera(pipeline, width=1280, height=720, fps=30):
    """配置DepthAI RGB相机节点"""
    # 创建RGB相机节点
    camera = pipeline.create(dai.node.ColorCamera)
    # 设置预览分辨率
    camera.setPreviewSize(width, height)
    # 设置传感器分辨率为1080p
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    # 设置非交错模式
    camera.setInterleaved(False)
    # 设置颜色顺序为BGR
    camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # 设置帧率
    camera.setFps(fps)

    # 创建输出节点并设置流名称为"rgb"
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    # 将相机预览输出连接到输出节点
    camera.preview.link(xout.input)

    return "rgb"

def get_gpu_memory():
    """获取当前GPU内存使用情况（已用/总计，单位MiB）"""
    try:
        # 初始化pynvml
        pynvml.nvmlInit()
        # 获取第一个GPU的句柄
        device = pynvml.nvmlDeviceGetHandleByIndex(0)
        # 获取内存信息
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(device)
        # 转换为MiB
        used_mem = mem_info.used // 1024 // 1024
        total_mem = mem_info.total // 1024 // 1024
        # 关闭pynvml
        pynvml.nvmlShutdown()
        return used_mem, total_mem
    except Exception as e:
        # 如果获取失败，记录警告并返回None
        logging.warning(f"无法获取GPU内存: {e}")
        return None, None

def _process_step(window_frames, is_first_step, model, grid_size, grid_query_frame, device, crop_width, crop_height):
    """处理一组裁剪帧，生成点跟踪预测"""
    try:
        # 将帧序列转换为张量，形状为(1, T, 3, H, W)
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=device
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )
        # 生成裁剪区域内的均匀网格点
        grid_points = []
        step_x = max(1, crop_width // grid_size)  # x方向步长
        step_y = max(1, crop_height // grid_size)  # y方向步长
        for i in range(grid_size):
            for j in range(grid_size):
                x = j * step_x + step_x // 2
                y = i * step_y + step_y // 2
                if 0 <= x < crop_width and 0 <= y < crop_height:
                    grid_points.append([x, y])
        if not grid_points:
            logging.warning("裁剪区域内未生成网格点")
            return None, None
        # 转换为张量，形状为(1, N, 2)
        grid_points = torch.tensor(grid_points, device=device).float()[None]
        # 为查询点添加帧索引，生成(t, x, y)格式，形状为(1, N, 3)
        queries = torch.cat(
            [torch.zeros(grid_points.shape[0], grid_points.shape[1], 1, device=device).float() + grid_query_frame, 
             grid_points],
            dim=2
        )
        # 调用CoTracker模型进行预测
        result = model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries,
            grid_query_frame=grid_query_frame,
        )
        return result
    except Exception as e:
        # 如果处理失败，记录错误并返回None
        logging.error(f"_process_step错误: {e}")
        return None, None

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/scaled_online.pth",
        help="CoTracker模型参数路径",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=10,
        help="网格大小",
    )
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="从此帧开始计算网格跟踪",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="相机分辨率宽度",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="相机分辨率高度",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="相机帧率",
    )
    parser.add_argument(
        "--crop_x1",
        type=int,
        default=600,
        help="裁剪区域左上角x坐标",
    )
    parser.add_argument(
        "--crop_y1",
        type=int,
        default=0,
        help="裁剪区域左上角y坐标",
    )
    parser.add_argument(
        "--crop_x2",
        type=int,
        default=850,
        help="裁剪区域右下角x坐标",
    )
    parser.add_argument(
        "--crop_y2",
        type=int,
        default=350,
        help="裁剪区域右下角y坐标",
    )

    # 解析参数
    args = parser.parse_args()

    # 验证裁剪坐标是否有效
    if not (0 <= args.crop_x1 < args.crop_x2 <= args.width and 0 <= args.crop_y1 < args.crop_y2 <= args.height):
        logging.error("裁剪坐标无效：必须在帧尺寸范围内")
        exit(1)

    # 初始化裁剪坐标
    crop_x1, crop_y1, crop_x2, crop_y2 = args.crop_x1, args.crop_y1, args.crop_x2, args.crop_y2
    crop_width = crop_x2 - crop_x1  # 裁剪宽度
    crop_height = crop_y2 - crop_y1  # 裁剪高度

    # 加载CoTracker模型
    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)  # 将模型移动到默认设备

    # 初始化DepthAI管道
    pipeline = dai.Pipeline()
    stream_name = setup_rgb_camera(pipeline, args.width, args.height, args.fps)

    # 初始化Visualizer，用于绘制跟踪轨迹
    vis = Visualizer(
        save_dir="./saved_videos",
        pad_value=120,
        linewidth=3,
    )

    # 启动DepthAI设备
    with dai.Device(pipeline) as device:
        # 获取RGB流队列
        queue = device.getOutputQueue(name=stream_name, maxSize=4, blocking=False)
        window_frames = []  # 存储帧序列
        is_first_step = True  # 标记是否为第一步
        frame_count = 0  # 帧计数
        pred_tracks = None  # 跟踪点坐标
        pred_visibility = None  # 点可见性
        prev_tracks = None  # 上一帧的跟踪点
        processing_times = []  # 每帧处理时间列表

        # 设置显示窗口大小为裁剪区域尺寸
        cv2.namedWindow("CoTracker DepthAI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CoTracker DepthAI", crop_width, crop_height)

        # 主循环
        while True:
            start_time = time.time()  # 记录帧处理开始时间

            # 尝试获取帧
            frame_data = queue.tryGet()
            if frame_data is not None:
                # 获取BGR帧并转换为RGB
                frame_bgr = frame_data.getCvFrame()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # 裁剪帧
                crop_rgb = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2, :]
                if crop_rgb.shape[0] != crop_height or crop_rgb.shape[1] != crop_width:
                    logging.warning("裁剪尺寸无效，跳过此帧")
                    continue

                window_frames.append(crop_rgb)  # 添加到帧序列
                frame_count += 1  # 帧计数增加

                # 当收集到足够帧时进行处理
                if frame_count % model.step == 0 and frame_count >= model.step:
                    if len(window_frames) < model.step * 2:
                        logging.warning(f"帧数量不足 ({len(window_frames)}/{model.step * 2})")
                        continue
                    # 处理帧序列，获取跟踪结果
                    pred_tracks, pred_visibility = _process_step(
                        window_frames,
                        is_first_step,
                        model,
                        args.grid_size,
                        args.grid_query_frame,
                        DEFAULT_DEVICE,
                        crop_width=crop_width,
                        crop_height=crop_height,
                    )
                    if is_first_step and (pred_tracks is None or pred_visibility is None):
                        logging.warning("第一步返回None，查询点可能未初始化")
                    is_first_step = False

                    # 限制帧序列长度，避免内存问题
                    window_frames = window_frames[-model.step * 2 :]

                # 可视化当前帧并更新裁剪坐标
                if pred_tracks is not None and pred_visibility is not None:
                    curr_tracks = pred_tracks[:, -1:, :, :]  # 当前帧跟踪点 (B, 1, N, 2)
                    curr_visibility = pred_visibility[:, -1:, :]  # 当前帧可见性 (B, 1, N)

                    # 计算点的位移
                    displacement = torch.zeros(2, device=DEFAULT_DEVICE)
                    if prev_tracks is not None:
                        visible_mask = curr_visibility[0, 0, :] > 0.5  # 可见点掩码
                        if visible_mask.sum() > 0:
                            # 计算可见点的位移
                            point_displacements = curr_tracks[0, 0, visible_mask, :] - prev_tracks[0, 0, visible_mask, :]
                            # 计算位移的L2范数
                            displacement_norms = torch.norm(point_displacements, dim=1)
                            moved_mask = displacement_norms > 0.1  # 移动点阈值
                            if moved_mask.sum() > 0:
                                # 取最大位移
                                displacement = point_displacements[moved_mask].max(dim=0).values

                    # 记录点位置和位移
                    curr_points = curr_tracks[0, 0, :, :].cpu().numpy()
                    logging.info(f"帧 {frame_count} 点位置: {curr_points.tolist()}")
                    logging.info(f"帧 {frame_count} 位移: {displacement.cpu().numpy().tolist()}")

                    # 根据位移更新裁剪坐标
                    crop_x1 += int(displacement[0].item())
                    crop_y1 += int(displacement[1].item())
                    crop_x2 = crop_x1 + crop_width
                    crop_y2 = crop_y1 + crop_height

                    # 确保裁剪坐标在帧边界内
                    crop_x1 = max(0, min(crop_x1, args.width - crop_width))
                    crop_y1 = max(0, min(crop_y1, args.height - crop_height))
                    crop_x2 = crop_x1 + crop_width
                    crop_y2 = crop_y1 + crop_height

                    # 将裁剪帧转换为张量，形状为(1, 1, 3, crop_height, crop_width)
                    video = torch.tensor(crop_rgb, device=DEFAULT_DEVICE).permute(2, 0, 1)[None, None]

                    # 在裁剪帧上绘制跟踪轨迹
                    res_video = vis.draw_tracks_on_video(
                        video=video,
                        tracks=curr_tracks,
                        visibility=curr_visibility,
                        query_frame=args.grid_query_frame,
                    )
                    # 转换为BGR格式以供OpenCV显示
                    res_frame = res_video[0, 0].permute(1, 2, 0).cpu().numpy()
                    res_frame_bgr = cv2.cvtColor(res_frame, cv2.COLOR_RGB2BGR)

                    # 更新上一帧的跟踪点
                    prev_tracks = curr_tracks.clone()
                else:
                    # 如果没有跟踪结果，直接显示裁剪帧
                    res_frame_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

                # 显示帧
                cv2.imshow("CoTracker DepthAI", res_frame_bgr)

                # 记录处理时间
                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                # 如果使用CUDA，记录GPU内存使用情况
                if DEFAULT_DEVICE == "cuda":
                    used_mem, total_mem = get_gpu_memory()
                    if used_mem is not None and total_mem is not None:
                        logging.info(f"GPU内存使用: {used_mem}/{total_mem} MiB")

                # 每100帧记录一次平均处理时间和FPS
                if frame_count % 100 == 0 and processing_times:
                    avg_processing_time = sum(processing_times) / len(processing_times)
                    avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                    logging.info(f"最近 {len(processing_times)} 帧平均处理时间: {avg_processing_time:.3f} 秒")
                    logging.info(f"最近 {len(processing_times)} 帧平均FPS: {avg_fps:.2f}")

                # 按'q'键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # 记录最终平均处理时间和FPS
                    if processing_times:
                        avg_processing_time = sum(processing_times) / len(processing_times)
                        avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                        logging.info(f"最终平均处理时间: {avg_processing_time:.3f} 秒")
                        logging.info(f"最终平均FPS: {avg_fps:.2f}")
                    break

        # 清理资源
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()