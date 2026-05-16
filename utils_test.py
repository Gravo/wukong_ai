import win32api
import time

def get_mouse_position():
    while True:
        x, y = win32api.GetCursorPos()
        print(f"Mouse position: X={x}, Y={y}")
        time.sleep(0.5)  # 每0.5秒更新一次坐标

if __name__ == '__main__':
    get_mouse_position()