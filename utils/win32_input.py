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




def attach1():
    # 点击鼠标右键（在当前鼠标位置）
    pydirectinput.click(button='left')
    return



def attach2():
    # 点击鼠标右键（在当前鼠标位置）
    pydirectinput.click(button='right')

    # # 在指定位置点击鼠标右键
    # pydirectinput.click(x, y, button='right')
    return


def collect_point():
    # 按下鼠标右键
    pydirectinput.mouseDown(button='right')

    # 保持按下状态4秒
    time.sleep(6.5)

    # 释放鼠标右键
    pydirectinput.mouseUp(button='right')


def random_action():
    import random
    keys = ['w', 's', 'a', 'd', ' ']
    return keys[random.randint(0,4)]


if __name__ == '__main__':
    #hwnd = win32gui.FindWindow(None, "无标题 - Notepad")
    hwnd = win32gui.FindWindow(None, "b1  ")

    if hwnd != 0:
        disable_ime(hwnd)
    else:
        print("窗口未找到。")

    
    #hwnd = win32gui.GetForegroundWindow()
    print(hwnd) #1640484

    time.sleep(3)
        
    collect_point()
    exit()


    while(True):
        # 发送 'A' 键
        #send_key_to_game('a', hwnd)
        key = random_action()
        pydirectinput.keyDown(key)
        pydirectinput.keyUp(key)
        hwnd = win32gui.GetForegroundWindow()
        print(hwnd)
        time.sleep(0.5)
        
        
    
    if hwnd == 0:
        print("未找到指定窗口。")
    else:
        try:
            # 发送 'A' 键
            send_key_to_game('a', hwnd)
            print("键发送成功。")
        except Exception as e:
            print(f"发送键失败：{e}")
