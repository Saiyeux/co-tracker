import ffmpeg
import math
import os

def frame_to_time(frame_number, fps):
    """将帧号转换为时间（分:秒）"""
    total_seconds = frame_number / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return minutes, seconds

def get_video_info(video_path):
    """获取视频的帧率和帧数"""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None, None
    
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            print("No video stream found in file")
            return None, None
        
        fps = eval(video_stream['r_frame_rate'])  # 获取帧率
        total_frames = int(video_stream['nb_frames'])  # 总帧数
        return fps, total_frames
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None

def main(video_path, target_frame):
    """主函数：计算目标帧对应的时间"""
    fps, total_frames = get_video_info(video_path)
    
    if fps is None or total_frames is None:
        print("Failed to get video info")
        return
    
    if target_frame > total_frames:
        print(f"Target frame {target_frame} exceeds total frames {total_frames}")
        return
    
    minutes, seconds = frame_to_time(target_frame, fps)
    print(f"Frame {target_frame} is at {minutes} minutes {seconds:.2f} seconds")

# 示例用法
video_path = "./assets/test.mp4"  # 替换为你的视频文件路径
target_frame = 10000  # 想查询的帧号

main(video_path, target_frame)