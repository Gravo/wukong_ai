import pydirectinput
import win32gui
import time
import win32con
from ctypes import windll



def disable_ime(hwnd):
    # 禁用IME输入法，临时的
    ImmDisableIME = windll.imm32.ImmDisableIME
    ImmDisableIME(hwnd)

    # 之后可以重新启用IME输入法
    # ImmEnableIME = windll.imm32.ImmEnableIME
    # ImmEnableIME(hwnd)


def send_key_to_game(key, hwnd):
    #获取到hwnd之后，建议保存
    win32gui.SetForegroundWindow(hwnd)
    pydirectinput.keyDown(key)
    pydirectinput.keyUp(key)





def collect_point():
    # 按下鼠标右键
    pydirectinput.mouseDown(button='right')

    # 保持按下状态4秒
    time.sleep(6.5)

    # 释放鼠标右键
    pydirectinput.mouseUp(button='right')


def restart_to_tiger():
    # hwnd = win32gui.FindWindow(None, w"无标题 - Notepad")
    hwnd = win32gui.FindWindow(None, "b1  ")
    if hwnd != 0:
        disable_ime(hwnd)
    else:
        print("窗口未找到。")
        exit()
    print(hwnd)  # 1640484
    win32gui.SetForegroundWindow(hwnd)
    # pydirectinput.keyDown('shiftleft')
    for i in range(4):
        pydirectinput.keyDown('d')
        pydirectinput.keyDown('w')
        time.sleep(0.3)
        pydirectinput.keyUp('d')
        pydirectinput.keyUp('w')
        time.sleep(0.01)
    pydirectinput.keyDown('d')
    time.sleep(0.4)
    pydirectinput.keyUp('d')
    for i in range(4):
        pydirectinput.keyDown('w')
        pydirectinput.keyDown('a')
        time.sleep(0.9)
        pydirectinput.keyUp('w')
        pydirectinput.keyUp('a')

        # pydirectinput.keyDown('d')
        # time.sleep(0.1)
        # pydirectinput.keyUp('d')
    pydirectinput.keyDown('w')
    time.sleep(3)
    pydirectinput.keyUp('w')
    pydirectinput.keyDown('a')
    time.sleep(0.6)
    pydirectinput.keyUp('a')
    pydirectinput.keyDown('w')
    time.sleep(5)
    pydirectinput.keyUp('w')

    for i in range(2): #4
        pydirectinput.keyDown('a')
        time.sleep(0.2)
        pydirectinput.keyUp('a')

        pydirectinput.keyDown('w')
        time.sleep(1)
        pydirectinput.keyUp('w')

        pydirectinput.keyDown('d')
        time.sleep(0.05)
        pydirectinput.keyUp('d')

        pydirectinput.keyDown('w')
        time.sleep(2)
        pydirectinput.keyUp('w')
    # pydirectinput.keyUp('shiftleft')


if __name__ == '__main__':
    restart_to_tiger()
