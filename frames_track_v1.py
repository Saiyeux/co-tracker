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

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor

logging.basicConfig(level=logging.INFO)

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

def setup_rgb_camera(pipeline, width=1280, height=720, fps=30):
    """Configure DepthAI RGB camera node."""
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

def _process_step(window_frames, is_first_step, model, grid_size, grid_query_frame, device):
    """Process a window of frames to get tracking predictions."""
    try:
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=device
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        print(f"video_chunk shape: {video_chunk.shape}")  # 调试输入形状
        result = model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )
        print(f"model output: {result}")  # 调试模型输出
        return result
    except Exception as e:
        print(f"Error in _process_step: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/scaled_online.pth",
        help="CoTracker model parameters",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=10,
        help="Regular grid size grid size",
    )
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera resolution width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Camera resolution height",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera frame rate",
    )

    args = parser.parse_args()

    # Load CoTracker model
    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)

    # Initialize DepthAI pipeline
    pipeline = dai.Pipeline()
    stream_name = setup_rgb_camera(pipeline, args.width, args.height, args.fps)

    # Initialize Visualizer
    vis = Visualizer(
        save_dir="./saved_videos",
        pad_value=120,
        linewidth=3,
        # fps=args.fps,
        # mode="rainbow",
    )

    # Start DepthAI device
    with dai.Device(pipeline) as device:
        queue = device.getOutputQueue(name=stream_name, maxSize=4, blocking=False)
        window_frames = []
        is_first_step = True
        frame_count = 0
        pred_tracks = None
        pred_visibility = None

        while True:
            frame_data = queue.tryGet()
            if frame_data is not None:
                # Get BGR frame and convert to RGB
                frame_bgr = frame_data.getCvFrame()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                window_frames.append(frame_rgb)
                frame_count += 1

                # Process frames when enough are collected
                if frame_count % model.step == 0 and frame_count >= model.step:
                    print(f"window_frames length: {len(window_frames)}")
                    print(f"frame shape: {window_frames[-1].shape}")
                    pred_tracks, pred_visibility = _process_step(
                        window_frames,
                        is_first_step,
                        model,
                        args.grid_size,
                        args.grid_query_frame,
                        DEFAULT_DEVICE,
                    )
                    is_first_step = False
                    if pred_tracks is not None and pred_visibility is not None:
                        print(f"pred_visibility shape: {pred_visibility.shape}")
                        print(f"pred_tracks shape: {pred_tracks.shape}")
                    else:
                        print("Warning: pred_tracks or pred_visibility is None")

                    # Limit window size to avoid memory issues
                    window_frames = window_frames[-model.step * 2 :]

                # Visualize current frame
                if pred_tracks is not None and pred_visibility is not None:
                    # Prepare single-frame video tensor
                    video = torch.tensor(frame_rgb, device=DEFAULT_DEVICE).permute(2, 0, 1)[None, None]  # (1, 1, 3, H, W)
                    # Adjust tracks and visibility for single frame
                    curr_tracks = pred_tracks[:, -1:, :, :]  # Take the latest frame's tracks
                    curr_visibility = pred_visibility[:, -1:, :]  # Take the latest frame's visibility

                    # Draw tracks on the current frame
                    res_video = vis.draw_tracks_on_video(
                        video=video,
                        tracks=curr_tracks,
                        visibility=curr_visibility,
                        query_frame=args.grid_query_frame,
                    )
                    # Convert result back to BGR for OpenCV display
                    res_frame = res_video[0, 0].permute(1, 2, 0).cpu().numpy()
                    res_frame_bgr = cv2.cvtColor(res_frame, cv2.COLOR_RGB2BGR)
                else:
                    res_frame_bgr = frame_bgr

                # Display the frame
                cv2.imshow("CoTracker DepthAI", res_frame_bgr)

                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Clean up
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()