# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np
import ffmpeg

from PIL import Image
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# 定义新的视频读取函数，使用 ffmpeg
def read_video_with_ffmpeg(video_path):
    """
    使用 ffmpeg 逐帧读取视频，返回 NumPy 数组，形状为 (帧数, 高度, 宽度, 通道)。
    """
    try:
        # 探测视频信息
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        
        # 使用 ffmpeg 读取视频帧
        process = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )
        
        frames = []
        while True:
            # 读取一帧的原始数据
            in_bytes = process.stdout.read(width * height * 3)  # RGB24 每像素 3 字节
            if not in_bytes:
                break
            # 转换为 NumPy 数组，形状 (height, width, 3)
            frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))
            frames.append(frame)
        
        process.stdout.close()
        process.wait()
        
        # 将帧列表转换为 NumPy 数组
        video = np.stack(frames, axis=0)  # 形状 (帧数, 高度, 宽度, 通道)
        return video
    
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise
    except Exception as e:
        print(f"Error reading video: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="/home/surgicalai/Data/test.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/apple_mask.png",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument(
        "--use_v2_model",
        action="store_true",
        help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Pass it if you would like to use the offline model",
    )

    args = parser.parse_args()

    # 使用 ffmpeg 加载视频
    video = read_video_with_ffmpeg(args.video_path)
    print('Video read success...')
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    print('Torch permute success...')
    segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
    print('Image open success...')
    segm_mask = torch.from_numpy(segm_mask)[None, None]
    print('Load video success...')

    if args.checkpoint is not None:
        if args.use_v2_model:
            model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
        else:
            if args.offline:
                window_len = 60
            else:
                window_len = 16
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=args.use_v2_model,
                offline=args.offline,
                window_len=window_len,
            )
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    print('Load model success...')
    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)
    print('Starting prediction...')
    pred_tracks, pred_visibility = model(
        video,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
        backward_tracking=args.backward_tracking,
        # segm_mask=segm_mask
    )
    print("computed")

    # 保存预测轨迹的视频
    seq_name = args.video_path.split("/")[-1]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
        query_frame=0 if args.backward_tracking else args.grid_query_frame,
    )
