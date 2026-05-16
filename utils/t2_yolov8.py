import torch
from ultralytics import YOLO
import cv2
import numpy as np
# 加载YOLOv8姿态估计模型
model = YOLO("yolov8n-pose.pt")  # 请确保你有这个模型文件
file_name = f"logs/screenshot_20240907_12-51-28.png"
model.info()

# 对图像进行预测
img = cv2.imread(file_name)
results = model(img)

# 获取检测到的边界框数量
num_objects = len(results)

# 打印检测到的对象数量
print(f"Detected {num_objects} objects.")


# 渲染检测结果

# 获取原始图像
orig_img = img

keypoints = results[0].keypoints.xy
if keypoints is not None and len(keypoints) > 0:

    # 绘制关键点
    for keypoint in keypoints:
        print(keypoint)
        if isinstance(keypoint, torch.Tensor):
            keypoint = keypoint.cpu().numpy()

            #遍历并绘制
            for point in keypoint:
                x, y = map(int, point)
                cv2.circle(orig_img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)


# # 检查是否检测到了关键点
# if results is not None and len(results) > 0:
#     # 遍历每个关键点
#     for r in results:
#         keypoints = r.keypoints
#         for idx, keypoint in enumerate(keypoints):
#             # keypoint format: (x, y, visibility)
#             print(keypoint)
#             # x, y, visibility = keypoint
#             # if visibility > 0.5:  # 只绘制可见的关键点
#             #     cv2.circle(orig_img, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)

# 从结果中获取关键点

# 显示渲染后的图像
cv2.imshow("YOLOv8 Pose Estimation", orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存渲染后的图像
cv2.imwrite("annotated_image.jpg", orig_img)

