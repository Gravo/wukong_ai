import cv2, numpy as np, dxcam

# 1. 截一张当前游戏画面
cam = dxcam.create(region=(0,0,1920,1080), output_color="BGR")
frame = cam.grab()
del cam
if frame is not None:
    resized = cv2.resize(frame, (224, 224))
    print(f"当前游戏画面: mean={resized.mean():.1f}, std={resized.std():.1f}")
    cv2.imwrite("diag_game_now.png", frame)

# 2. 看训练数据
data = np.load("pathfinding_data/preprocessed/stacked_data.npz", allow_pickle=False)
train_frames = data["frames"]  # (N, 12, 224, 224) uint8
# 取第一帧的第一个stack
first = train_frames[0, :3].transpose(1, 2, 0)  # (224, 224, 3) RGB
first_bgr = first[:, :, ::-1].copy()
print(f"训练数据帧: mean={first_bgr.mean():.1f}, std={first_bgr.std():.1f}")
cv2.imwrite("diag_train_sample.png", first_bgr)

# 3. 对比
print(f"\n对比:")
print(f"  当前游戏: mean={resized.mean():.1f}")
print(f"  训练数据: mean={first_bgr.mean():.1f}")
