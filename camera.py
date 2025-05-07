#!/usr/bin/env python3
import cv2
import numpy as np
import depthai as dai
import time
import os
import logging

logging.basicConfig(level=logging.INFO)

def setup_rgb_camera(pipeline, width=1280, height=720, fps=30):
    """配置DepthAI RGB相机节点"""
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

def main():
    output_dir = os.path.join(os.getcwd(), 'img')
    os.makedirs(output_dir, exist_ok=True)

    pipeline = dai.Pipeline()
    stream_name = setup_rgb_camera(pipeline)

    with dai.Device(pipeline) as device:
        queue = device.getOutputQueue(name=stream_name, maxSize=4, blocking=False)
        fps_counter = 0
        fps_time = time.time()
        
        while True:
            frame_data = queue.tryGet()
            if frame_data is not None:
                frame = frame_data.getCvFrame()
                resized_frame = cv2.resize(frame, (1280, 720))
                cv2.imshow('RGB Camera', resized_frame)
                
                fps_counter += 1

            if time.time() - fps_time >= 1.0:
                scale = time.time() - fps_time
                fps = fps_counter / scale
                print(f"FPS: {fps:.2f}")
                fps_counter = 0
                fps_time = time.time()

            key = cv2.waitKey(1)

            if key == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()