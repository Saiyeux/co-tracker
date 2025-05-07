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

from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def draw_rectangle(rgb, top_left, bottom_right, color=(255, 0, 0), width=2):
    """Draw a rectangle on the frame."""
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [top_left, bottom_right],
        outline=color,
        width=width
    )
    return np.array(img)

def get_gpu_memory():
    """Get current GPU memory usage using pynvml (returns used/total in MiB)."""
    try:
        pynvml.nvmlInit()
        device = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use first GPU
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(device)
        used_mem = mem_info.used // 1024 // 1024  # Convert to MiB
        total_mem = mem_info.total // 1024 // 1024  # Convert to MiB
        pynvml.nvmlShutdown()
        return used_mem, total_mem
    except Exception as e:
        logging.warning(f"Failed to get GPU memory: {e}")
        return None, None

def _process_step(window_frames, is_first_step, model, grid_size, grid_query_frame, device, crop_coords=None):
    """Process a window of frames to get tracking predictions for cropped region."""
    try:
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=device
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)

        # If crop coordinates are provided, create a grid within the crop region
        if crop_coords:
            x1, y1, x2, y2 = crop_coords
            grid_points = []
            step_x = max(1, (x2 - x1) // grid_size)  # Ensure step_x is at least 1
            step_y = max(1, (y2 - y1) // grid_size)  # Ensure step_y is at least 1
            for i in range(grid_size):
                for j in range(grid_size):
                    x = x1 + j * step_x + step_x // 2
                    y = y1 + i * step_y + step_y // 2
                    if x1 <= x < x2 and y1 <= y < y2:  # Strict inequality to avoid boundary issues
                        grid_points.append([x, y])
            if not grid_points:
                logging.warning("No grid points generated in crop region")
                return None, None
            grid_points = torch.tensor(grid_points, device=device).float()[None]  # (1, N, 2)
            # Add frame index for queries (t, x, y)
            queries = torch.cat(
                [torch.zeros(grid_points.shape[0], grid_points.shape[1], 1, device=device).float() + grid_query_frame, 
                 grid_points],
                dim=2
            )  # (1, N, 3)
            result = model(
                video_chunk,
                is_first_step=is_first_step,
                queries=queries,
                grid_query_frame=grid_query_frame,
            )
        else:
            result = model(
                video_chunk,
                is_first_step=is_first_step,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
            )
        return result
    except Exception as e:
        logging.error(f"Error in _process_step: {e}")
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
        help="Regular grid size",
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
    parser.add_argument(
        "--crop_x1",
        type=int,
        default=600,
        help="Crop region x1 coordinate (top-left)",
    )
    parser.add_argument(
        "--crop_y1",
        type=int,
        default=0,
        help="Crop region y1 coordinate (top-left)",
    )
    parser.add_argument(
        "--crop_x2",
        type=int,
        default=850,
        help="Crop region x2 coordinate (bottom-right)",
    )
    parser.add_argument(
        "--crop_y2",
        type=int,
        default=350,
        help="Crop region y2 coordinate (bottom-right)",
    )

    args = parser.parse_args()

    # Validate crop coordinates
    if not (0 <= args.crop_x1 < args.crop_x2 <= args.width and 0 <= args.crop_y1 < args.crop_y2 <= args.height):
        logging.error("Invalid crop coordinates: must be within frame dimensions")
        exit(1)

    crop_coords = (args.crop_x1, args.crop_y1, args.crop_x2, args.crop_y2)

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
    )

    # Start DepthAI device
    with dai.Device(pipeline) as device:
        queue = device.getOutputQueue(name=stream_name, maxSize=4, blocking=False)
        window_frames = []
        is_first_step = True
        frame_count = 0
        pred_tracks = None
        pred_visibility = None
        box_offset = torch.tensor([args.crop_x1, args.crop_y1], device=DEFAULT_DEVICE).float()
        processing_times = []  # List to store per-frame processing times

        while True:
            start_time = time.time()  # Record start time for frame processing

            frame_data = queue.tryGet()
            if frame_data is not None:
                # Get BGR frame and convert to RGB
                frame_bgr = frame_data.getCvFrame()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                window_frames.append(frame_rgb)
                frame_count += 1

                # Process frames when enough are collected
                if frame_count % model.step == 0 and frame_count >= model.step:
                    if len(window_frames) < model.step * 2:
                        logging.warning(f"Insufficient frames ({len(window_frames)}/{model.step * 2})")
                        continue
                    pred_tracks, pred_visibility = _process_step(
                        window_frames,
                        is_first_step,
                        model,
                        args.grid_size,
                        args.grid_query_frame,
                        DEFAULT_DEVICE,
                        crop_coords=crop_coords,
                    )
                    if is_first_step and (pred_tracks is None or pred_visibility is None):
                        logging.warning("First step returned None, queries may not be initialized")
                    is_first_step = False

                    # Limit window size to avoid memory issues
                    window_frames = window_frames[-model.step * 2 :]

                # Visualize current frame
                if pred_tracks is not None and pred_visibility is not None:
                    curr_tracks = pred_tracks[:, -1:, :, :]  # (B, 1, N, 2)
                    curr_visibility = pred_visibility[:, -1:, :]  # (B, 1, N)
                    if is_first_step:
                        initial_tracks = pred_tracks[:, 0:1, :, :]  # (B, 1, N, 2)
                    else:
                        initial_tracks = pred_tracks[:, args.grid_query_frame:args.grid_query_frame+1, :, :]  # (B, 1, N, 2)

                    # Calculate average displacement of visible points
                    visible_mask = curr_visibility[0, 0, :] > 0.5  # (N,)
                    if visible_mask.sum() > 0:
                        displacement = (
                            curr_tracks[0, 0, visible_mask, :] - 
                            initial_tracks[0, 0, visible_mask, :]
                        ).mean(dim=0)  # (2,)
                    else:
                        displacement = torch.zeros(2, device=DEFAULT_DEVICE)

                    # Update box position
                    curr_box_top_left = (box_offset + displacement).int()
                    curr_box_bottom_right = (
                        curr_box_top_left[0] + (args.crop_x2 - args.crop_x1),
                        curr_box_top_left[1] + (args.crop_y2 - args.crop_y1)
                    )

                    # Ensure box stays within frame
                    curr_box_top_left = (
                        max(0, min(curr_box_top_left[0].item(), args.width - (args.crop_x2 - args.crop_x1))),
                        max(0, min(curr_box_top_left[1].item(), args.height - (args.crop_y2 - args.crop_y1)))
                    )
                    curr_box_bottom_right = (
                        curr_box_top_left[0] + (args.crop_x2 - args.crop_x1),
                        curr_box_top_left[1] + (args.crop_y2 - args.crop_y1)
                    )

                    # Draw rectangle on frame
                    frame_rgb_with_box = draw_rectangle(
                        frame_rgb,
                        curr_box_top_left,
                        curr_box_bottom_right,
                        color=(255, 0, 0),
                        width=2
                    )

                    # Prepare single-frame video tensor with box
                    video = torch.tensor(frame_rgb_with_box, device=DEFAULT_DEVICE).permute(2, 0, 1)[None, None]  # (1, 1, 3, H, W)

                    # Adjust tracks for visualization
                    vis_tracks = pred_tracks  # Directly use pred_tracks as they are in original frame coordinates

                    # Draw tracks on the current frame
                    res_video = vis.draw_tracks_on_video(
                        video=video,
                        tracks=vis_tracks[:, -1:, :, :],
                        visibility=curr_visibility,
                        query_frame=args.grid_query_frame,
                    )
                    # Convert result back to BGR for OpenCV display
                    res_frame = res_video[0, 0].permute(1, 2, 0).cpu().numpy()
                    res_frame_bgr = cv2.cvtColor(res_frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_rgb_with_box = draw_rectangle(
                        frame_rgb,
                        (args.crop_x1, args.crop_y1),
                        (args.crop_x2, args.crop_y2),
                        color=(255, 0, 0),
                        width=2
                    )
                    res_frame_bgr = cv2.cvtColor(frame_rgb_with_box, cv2.COLOR_RGB2BGR)

                # Display the frame
                cv2.imshow("CoTracker DepthAI", res_frame_bgr)

                # Log processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                logging.info(f"Frame {frame_count} processing time: {processing_time:.3f} seconds")

                # Log GPU memory usage if on CUDA
                if DEFAULT_DEVICE == "cuda":
                    used_mem, total_mem = get_gpu_memory()
                    if used_mem is not None and total_mem is not None:
                        logging.info(f"GPU memory usage: {used_mem}/{total_mem} MiB")

                # Calculate and log average FPS every 100 frames
                if frame_count % 100 == 0 and processing_times:
                    avg_processing_time = sum(processing_times) / len(processing_times)
                    avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                    logging.info(f"Average FPS (last {len(processing_times)} frames): {avg_fps:.2f}")

                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # Log final average FPS
                    if processing_times:
                        avg_processing_time = sum(processing_times) / len(processing_times)
                        avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                        logging.info(f"Final average FPS: {avg_fps:.2f}")
                    break

        # Clean up
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()