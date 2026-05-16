import cv2
import os

# 视频文件路径
video_path = 'Video_2024-09-29_225550.mp4'
# 输出目录
output_dir = 'data_video'
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 目标分辨率
target_width = 1280
target_height = 720

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
# 每两分钟的帧数
frames_per_two_minutes = int(fps * 60 * 2)

# 初始化计数器
frame_count = 0
segment_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 如果没有更多帧，退出循环

    # 调整图像大小
    resized_frame = cv2.resize(frame, (target_width, target_height))

    # 计算当前段的目录
    segment_dir = os.path.join(output_dir, f'segment_{segment_count:03d}')
    os.makedirs(segment_dir, exist_ok=True)

    # 保存当前帧
    frame_filename = os.path.join(segment_dir, f'frame_{frame_count:06d}.jpg')
    cv2.imwrite(frame_filename, resized_frame)

    # 更新计数器
    frame_count += 1

    # 每两分钟创建一个新的段
    if frame_count % frames_per_two_minutes == 0:
        segment_count += 1

# 释放视频捕获对象
cap.release()

print("Video processing complete.")