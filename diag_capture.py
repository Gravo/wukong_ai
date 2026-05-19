import time, cv2, dxcam, numpy as np

cam = dxcam.create(region=(0,0,1920,1080), output_color="BGR")
print("截图5次，检查内容...")
for i in range(5):
    frame = cam.grab()
    if frame is not None:
        print(f"  frame {i}: shape={frame.shape}, mean={frame.mean():.1f}, min={frame.min()}, max={frame.max()}")
        # 保存一帧看看
        if i == 0:
            cv2.imwrite("diag_frame.png", frame)
            print(f"  已保存 diag_frame.png")
    time.sleep(0.5)
del cam

# 也试试mss
print("\n用mss截图:")
import mss
sct = mss.mss()
monitor = {"top":0,"left":0,"width":1920,"height":1080}
shot = sct.grab(monitor)
frame2 = np.array(shot)[:,:,:3]
print(f"  mss: shape={frame2.shape}, mean={frame2.mean():.1f}, min={frame2.min()}, max={frame2.max()}")
cv2.imwrite("diag_frame_mss.png", frame2)
print(f"  已保存 diag_frame_mss.png")
sct.close()
