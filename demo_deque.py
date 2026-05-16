import time

import cv2
from collections import deque
from grabscreen import *
# 初始化一个双端队列作为缓存
frame_cache = deque(maxlen=12)
window_size = (224, 36, 800, 556)  # 384,344  192,172 96,86


while True:
    # 捕获屏幕图像
    screen = grab_screen(window_size)

    # 将图像转换为灰度
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    # 将灰度图像添加到缓存中
    frame_cache.append(screen_gray)

    # 如果缓存已满（即包含12帧）
    if len(frame_cache) == 12:
        # 获取最近的4帧
        recent_frames = list(frame_cache)[-4:]

        # 在这里对这四帧进行堆叠或其他操作...
        # ...
        import numpy as np
        import cv2

        # 假设 recent_frames 包含了4个灰度帧
        # recent_frames = [frame1, frame2, frame3, frame4]

        # 检查所有帧的形状是否一致
        assert all(f.shape == recent_frames[0].shape for f in recent_frames), "All frames must have the same shape."

        # 将前两个帧水平拼接
        top_row = np.hstack(recent_frames[:2])

        # 将后两个帧水平拼接
        bottom_row = np.hstack(recent_frames[2:])

        # 将两行垂直拼接
        combined_frame = np.vstack([top_row, bottom_row])
        WIDTH = 96 * 2
        HEIGHT = 88 * 2

        station = cv2.resize(combined_frame,(WIDTH,HEIGHT))
        # 显示拼接后的图像
        cv2.imshow('Combined Frames', station)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()
        time.sleep(0.01)