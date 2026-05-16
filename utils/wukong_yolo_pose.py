import cv2
import numpy as np
from ultralytics import YOLO
import pygetwindow as gw
from PIL import ImageGrab
from datetime import datetime

# 获取指定窗口的图像
def get_window_image(window_title):
    window = gw.getWindowsWithTitle(window_title)[0]
    if window:
        window.activate()
        window.moveTo(0, 0)
        #window.resizeTo(800, 600)  # 假设我们知道窗口的大小

        # 获取窗口的位置和尺寸
        left, top, right, bottom = window.left, window.top, window.right, window.bottom
        print(left, top, right, bottom)
        # 使用Pillow的ImageGrab来捕获窗口截图
        screenshot = ImageGrab.grab(bbox=(left + 20, top + 50, right, bottom))
       
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    else:
        print("Window not found")
        return None

# 加载YOLOv8姿态估计模型
model = YOLO("yolov8n-pose.pt")  # 请确保你有这个模型文件

# 指定窗口标题
window_title = "b1  "

# 获取窗口图像
frame = get_window_image(window_title)

if frame is not None:
    # 使用YOLOv8进行姿态估计
    results = model(frame)
    
    # 处理结果
    for *xyxy, conf, cls in results.xyxy[0].numpy():
        # xyxy是边界框坐标，conf是置信度，cls是类别
        print(f"Confidence: {conf}, Class: {cls}")
    
    # 可视化结果
    annotated_frame = results.render()[0]
    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
