import pygetwindow as gw
from PIL import ImageGrab
from datetime import datetime
from ultralytics import YOLO
import cv2
import numpy as np
import time


# 加载YOLOv8姿态估计模型
model = YOLO("yolov8n-pose.pt")  # 请确保你有这个模型文件

# 指定窗口标题
window_title = "b1  "
window = gw.getWindowsWithTitle(window_title)[0]
window.activate()
window.moveTo(0, 0)
# 获取窗口的位置和尺寸
left, top, right, bottom = window.left, window.top, window.right, window.bottom
print(left, top, right, bottom)

while(True):
    
    # 使用Pillow的ImageGrab来捕获窗口截图
    screenshot = ImageGrab.grab(bbox=(left + 20, top + 50, right, bottom))

    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H-%M-%S")

        
    file_name = f"logs/screenshot_{date_time_str}.png"
    # 保存截图到文件
    screenshot.save(file_name)
    

    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 定义新的图像尺寸
    width = 300
    height = 200

    # 缩放图像
    resized_frame = frame
    # resized_frame = cv2.resize(frame, (width, height))

    # 显示缩放后的图像
    init = False
    try:        
        t1 = "YOLOv8 input"
        cv2.imshow(t1, resized_frame)

        
        if init == False:
            # 等待键盘输入，参数是等待时间（毫秒）
            # 如果按下 'q' 键，则退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            w1 = gw.getWindowsWithTitle(t1)[0]
            w1.moveTo(1100, 0)
            init = True
        
    except Exception as e:
        print(f"handle Yolov8 input fail：{e}")


    if resized_frame is not None:
        # 使用YOLOv8进行姿态估计
        results = model(resized_frame)

        # 显示缩放后的图像
        init_pos = False

        try:   
          
            # 可视化结果
            annotated_frame = results.render()

            t2 = "YoLov8 pose"
            cv2.imshow(t2,annotated_frame)
            if init_pos == False:
                # 等待键盘输入，参数是等待时间（毫秒）
                # 如果按下 'q' 键，则退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                w2 = gw.getWindowsWithTitle(t2)[0]
                w1.moveTo(1100, 300)
                # 将窗口设置为最前面
                cv2.setWindowProperty(t1, cv2.WND_PROP_TOPMOST, 1)
                init_pos = True
                
        except Exception as e:
            print(f"show pose estimation fail：{e}")

    
    
    
    time.sleep(0.02)





