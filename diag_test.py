import time, ctypes
from ctypes import wintypes

print("=== 1. 查找游戏窗口 ===")
user32 = ctypes.windll.user32
WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
windows = []
def enum_cb(hwnd, lparam):
    if user32.IsWindowVisible(hwnd):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        if buf.value:
            windows.append((hwnd, buf.value))
    return True
user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

game_w = [(h,t) for h,t in windows if any(k in t.lower() for k in ['黑神话','wukong','b1'])]
if game_w:
    for h,t in game_w:
        print(f"  找到: hwnd={h}, title=\"{t}\"")
else:
    print("  未找到游戏窗口！窗口列表:")
    for h,t in windows[:20]:
        print(f"    \"{t}\"")

print("\n=== 2. 截图测试 ===")
import dxcam
cam = dxcam.create(region=(0,0,1920,1080), output_color="BGR")
frame = cam.grab()
if frame is not None:
    print(f"  截图OK: shape={frame.shape}, mean={frame.mean():.1f}")
else:
    print("  截图失败!")
del cam

print("\n=== 3. 按键测试 ===")
import pydirectinput
print("  3秒后按A键(向左移动)...")
time.sleep(3)
pydirectinput.press("a")
print("  A键已发送!")

print("\n=== 4. 鼠标移动测试 ===")
print("  2秒后鼠标右移50px...")
time.sleep(2)
pydirectinput.moveRel(50, 0, relative=True)
print("  鼠标移动已发送!")

print("\n=== 诊断完成 ===")
