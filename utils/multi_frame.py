import numpy as np
from mss import mss
import cv2

import os

# 定义窗口大小
window_size = (224, 36, 800, 556)  # (left, top, width, height)

# 设置保存最近60帧的画面数组
frames = [None] * 60
frame_count = 0  # 记录当前帧的位置
merge_counter = 0  # 合并计数器
save_dir = './screenshots'  # 图片保存目录

os.makedirs(save_dir, exist_ok=True)  # 创建保存目录


def capture_and_preprocess():
    global frames, frame_count, merge_counter

    with mss() as sct:
        # 获得屏幕指定区域的图像
        monitor = {'top': window_size[1], 'left': window_size[0],
                   'width': window_size[2], 'height': window_size[3]}
        sct_img = sct.grab(monitor)

        # 转换为NumPy数组
        img = np.array(sct_img)

        # 转换为灰度图像
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        WIDTH = 144
        HEIGHT = 132

        # 缩放图像
        resized_img = cv2.resize(gray_img, (144, 132))  # 调整为模型输入尺寸

        # 归一化图像
        normalized_img = resized_img / 255.0
        # normalized_img = resized_img

        # 更新帧数组
        frames[frame_count % 60] = normalized_img
        frame_count += 1

        # 检查是否有足够的帧进行合并操作
        if all(frame is not None for frame in frames):
            # 每四帧进行一次合并操作
            if frame_count % 4 == 0:
                merge_counter += 1
                display_and_save_2x2(merge_counter)


def display_and_save_2x2(counter):
    # 获取指定位置的画面帧
    def get_frame(index):
        if index >= 0 and index < 60:
            return frames[index]
        else:
            return frames[index % 60]

    # 获取指定位置的四个画面帧
    frame1 = get_frame(counter - 1)
    frame2 = get_frame(counter - 2)
    frame3 = get_frame(counter - 3)
    frame4 = get_frame(counter - 4)

    # 检查所有图像是否具有相同的高度和宽度
    if any(frame is None for frame in [frame1, frame2, frame3, frame4]):
        raise ValueError("One or more frames are None.")

    # 水平拼接图像
    top_row = np.hstack((frame1, frame2))
    bottom_row = np.hstack((frame3, frame4))

    # 垂直拼接图像
    final_image = np.vstack((top_row, bottom_row))

    # 展示图像
    cv2.imshow('Merged Image', final_image)
    cv2.waitKey(1)

    # 保存图像
    save_path = os.path.join(save_dir, f'screenshot_2x2_{counter}.png')
    cv2.imwrite(save_path, final_image * 255)  # 将归一化后的图像恢复原样再保存

    # # 将图像转换为Tensor
    # tensor_image = torch.tensor(final_image, dtype=torch.float32).unsqueeze(0)  # 添加通道维度
    #
    # # 返回Tensor图像
    # return tensor_image


if __name__ == "__main__":
    while True:
        capture_and_preprocess()
        # 每次捕获之间稍作等待
        import time

        time.sleep(0.1)