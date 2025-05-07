# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np
import time
import glob  # 新增：用于读取图像文件列表

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",  # 修改：从 video_path 改为 image_dir
        default="/home/surgicalai/Data/images_5/",  # 图像文件夹路径
        help="path to a directory containing images",
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/scaled_online.pth",
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

    args = parser.parse_args()

    # 检查图像文件夹是否存在
    if not os.path.isdir(args.image_dir):
        raise ValueError("Image directory does not exist")

    # 获取图像文件列表并排序（假设文件名按顺序命名，如 frame_001.jpg）
    image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) + 
                         glob.glob(os.path.join(args.image_dir, "*.png")))
    if not image_files:
        raise ValueError("No images found in the directory")

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)

    window_frames = []

    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )

    # 逐帧读取图像，处理窗口
    is_first_step = True
    for i, image_path in enumerate(image_files):
        start_time = time.time()
        frame = iio.imread(image_path)  # 读取单帧图像
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
            )
            is_first_step = False
        processing_time = time.time() - start_time
        print(f"Processing frame {i + 1}/{len(image_files)}: {image_path} (Time: {processing_time:.4f}s)")
        window_frames.append(frame)
    
    # 处理最后一批帧
    if window_frames:
        pred_tracks, pred_visibility = _process_step(
            window_frames[-(i % model.step) - model.step - 1 :] if i >= model.step else window_frames,
            is_first_step,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
        )

    print("Tracks are computed")

    # 保存可视化视频
    seq_name = os.path.basename(args.image_dir)  # 使用文件夹名作为序列名
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame
    )