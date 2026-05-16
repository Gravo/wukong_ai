import cv2
from ultralytics import YOLO
import torch
from PIL import ImageGrab
import numpy as np
import pygetwindow as gw

# 加载模型
model = YOLO('yolov8n-pose.pt')  # 加载预训练的 YOLOv8 Pose 模型

connections = [
    (0, 1),  # 鼻子 -> 左眼
    (0, 2),  # 鼻子 -> 右眼
    (1, 3),  # 左眼 -> 左耳
    (2, 4),  # 右眼 -> 右耳
    (0, 5),  # 鼻子 -> 左肩
    (0, 6),  # 鼻子 -> 右肩
    (5, 7),  # 左肩 -> 左肘
    (7, 9),  # 左肘 -> 左腕
    (6, 8),  # 右肩 -> 右肘
    (8, 10),  # 右肘 -> 右腕
    (5, 6),  # 左肩 -> 右肩
    (5, 11),  # 左肩 -> 左髋
    (6, 12),  # 右肩 -> 右髋
    (11, 12),  # 左髋 -> 右髋
    (11, 13),  # 左髋 -> 左膝
    (13, 15),  # 左膝 -> 左踝
    (12, 14),  # 右髋 -> 右膝
    (14, 16)  # 右膝 -> 右踝
]



connections = [
    (0, 1),  # 鼻子 -> 左眼
    (0, 2),  # 鼻子 -> 右眼
    (11, 12),  # 左髋 -> 右髋
    (11, 13),  # 左髋 -> 左膝
    (13, 15),  # 左膝 -> 左踝
    (12, 14),  # 右髋 -> 右膝
    (14, 16)  # 右膝 -> 右踝
]

def pose_from_cam():
    global keypoints
    # 打开摄像头或加载视频文件
    cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
    # 定义关键点连接规则

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 使用模型进行预测
        results = model(frame)

        # 从结果中获取关键点
        keypoints = results[0].keypoints.xy
        if keypoints is not None and len(keypoints) > 0:
            # 绘制关键点
            for keypoint in keypoints:
                # 检查 keypoint 是否是一个包含多个点的张量
                if isinstance(keypoint, torch.Tensor) and keypoint.shape == (17, 2):
                    # 将张量转换为 NumPy 数组
                    keypoint = keypoint.cpu().numpy()

                    # 绘制每个点
                    for point in keypoint:
                        x, y = map(int, point)
                        cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

                    # 绘制连线
                    for connection in connections:
                        pt1, pt2 = connection
                        if pt1 < len(keypoint) and pt2 < len(keypoint):
                            point1 = keypoint[pt1]
                            point2 = keypoint[pt2]
                            cv2.line(frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])),
                                     (0, 255, 0),
                                     thickness=2)
                            # cv2.circle(frame, (int(point1[0]), int(point1[1])), 8, (0, 0, 255), thickness=-1,
                            #            lineType=cv2.FILLED)

        # 显示结果
        cv2.imshow('YOLOv8 Pose Estimation', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


# pose_from_cam()


def pose_from_window():
    import numpy as np
    import cv2
    from PIL import ImageGrab

    # 定义新的图像尺寸
    width = 320
    height = 240

    # 创建一个全零的图像，尺寸为 640x480，通道数为3（彩色图像）
    blank_image = np.zeros((240, 320, 3), np.uint8)
    t1 = 'YOLOv8 Pose Estimation'

    # 创建一个窗口，并设置窗口的名字
    cv2.namedWindow(t1, cv2.WINDOW_NORMAL)
    # 显示空白图像
    cv2.imshow(t1, blank_image)
    w1 = gw.getWindowsWithTitle(t1)[0]

    w1.moveTo(1000, 0)
    cv2.resizeWindow(t1, width, height)
    # 将窗口设置为最前面
    cv2.setWindowProperty(t1, cv2.WND_PROP_TOPMOST, 1)


    # 如果按下 'q' 键，则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()


    while True:
        # 使用PIL的ImageGrab捕获屏幕的一个区域
        bb_x = [40,180,1180,880]
        bb_x = [40,180,1180,880]

        image = ImageGrab.grab(bbox=bb_x)  # bbox 为可选参数，指定屏幕的坐标范围
        # # 将PIL图像转换为OpenCV图像
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        #
        # 获取图像的高度和宽度
        height, width = image.shape[:2]

        # 定义要转换为灰度模式的区域
        x_start = (width - 160) // 2  # 计算区域的起始x坐标
        y_start = height - 500  # 从底部开始向上200像素
        x_end = x_start + 200  # 区域宽度为80像素
        y_end = y_start + 300  # 区域高度为200像素

        # 提取该区域
        roi = image[y_start:y_end, x_start:x_end]

        # # 将该区域转换为灰度模式
        # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # # 将灰度区域转换回BGR格式，以便可以放回原图
        # gray_roi_bgr = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
        # # 将灰度区域重新应用到原图上
        # image[y_start:y_end, x_start:x_end] = gray_roi_bgr

        # 应用马赛克效果

        def apply_mosaic_effect(image, block_size=20):
            small = cv2.resize(roi, (roi.shape[1] // block_size, roi.shape[0] // block_size))

            large = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            return large

        # 对区域应用马赛克效果
        mosaic_roi = apply_mosaic_effect(roi)
        # 将马赛克区域重新应用到原图上
        image[y_start:y_end, x_start:x_end] = mosaic_roi


        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)




        # 缩放图像
        resized_frame = cv2.resize(frame, (width, height))
        # resized_frame = frame

        # 使用模型进行预测
        results = model(resized_frame)

        # 从结果中获取关键点
        keypoints = results[0].keypoints.xy
        if keypoints is not None and len(keypoints) > 0:
            # 绘制关键点
            for keypoint in keypoints:
                # 检查 keypoint 是否是一个包含多个点的张量
                if isinstance(keypoint, torch.Tensor) and keypoint.shape == (17, 2):
                    # 将张量转换为 NumPy 数组
                    keypoint = keypoint.cpu().numpy()

                    # 绘制每个点
                    for point in keypoint:
                        x, y = map(int, point)
                        cv2.circle(resized_frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

                    # 绘制连线
                    for connection in connections:
                        pt1, pt2 = connection
                        if pt1 < len(keypoint) and pt2 < len(keypoint):
                            point1 = keypoint[pt1]
                            point2 = keypoint[pt2]
                            cv2.line(resized_frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])),
                                     (0, 255, 0),
                                     thickness=2)

        # 显示结果
        cv2.imshow('YOLOv8 Pose Estimation', resized_frame)



        # 如果按下 'q' 键，则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源和关闭所有窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("here")
    pose_from_window()