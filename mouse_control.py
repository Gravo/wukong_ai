"""
mouse_control.py - 鼠标控制解决方案

尝试多种方法让游戏响应鼠标输入：
1. pydirectinput.moveRel (relative=True)
2. 平滑移动（分步移动）
3. SendInput API 直接调用
4. ctypes 直接调用 Windows API
"""
import time
import ctypes
from ctypes import wintypes


# ============================================================
# 方法 1: pydirectinput (最简单)
# ============================================================
def pydirectinput_move(dx, dy):
    """使用 pydirectinput 移动鼠标"""
    import pydirectinput as pdi
    pdi.move(dx, dy, relative=True)


# ============================================================
# 方法 2: 平滑移动 (更自然)
# ============================================================
def smooth_move(dx, dy, steps=10, duration=0.1):
    """
    分步平滑移动鼠标，模拟更自然的人类操作

    Args:
        dx: 总水平移动量（正=右，负=左）
        dy: 总垂直移动量（正=下，负=上）
        steps: 分步数
        duration: 总持续时间（秒）
    """
    import pydirectinput as pdi

    step_x = dx / steps
    step_y = dy / steps
    delay = duration / steps

    for _ in range(steps):
        pdi.moveRel(int(step_x), int(step_y), relative=True, duration=0)
        time.sleep(delay)


# ============================================================
# 方法 3: SendInput API (绕过 pydirectinput)
# ============================================================
def sendinput_move(dx, dy):
    """
    直接调用 Windows SendInput API 移动鼠标

    Args:
        dx: 水平移动量（正=右，负=左）
        dy: 垂直移动量（正=下，负=上）
    """
    MOUSEEVENTF_MOVE = 0x0001
    INPUT_MOUSE = 0

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
        ]

    class INPUT(ctypes.Structure):
        _fields_ = [
            ("type", wintypes.DWORD),
            ("mi", MOUSEINPUT),
        ]

    extra = ctypes.c_ulong(0)
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.dx = dx
    inp.mi.dy = dy
    inp.mi.mouseData = 0
    inp.mi.dwFlags = MOUSEEVENTF_MOVE
    inp.mi.time = 0
    inp.mi.dwExtraInfo = ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p)

    user32 = ctypes.windll.user32
    result = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if result != 1:
        print(f"[Warning] SendInput failed: {ctypes.GetLastError()}")
    return result


# ============================================================
# 方法 4: 平滑 SendInput (最平滑)
# ============================================================
def smooth_sendinput_move(dx, dy, steps=10, duration=0.1):
    """
    使用 SendInput API 实现平滑移动

    Args:
        dx: 总水平移动量
        dy: 总垂直移动量
        steps: 分步数
        duration: 总持续时间（秒）
    """
    step_x = dx / steps
    step_y = dy / steps
    delay = duration / steps

    for _ in range(steps):
        sendinput_move(int(step_x), int(step_y))
        time.sleep(delay)


# ============================================================
# 测试函数
# ============================================================
def test_all_methods():
    """测试所有鼠标控制方法"""
    print("=" * 60)
    print("鼠标控制方法测试")
    print("=" * 60)
    print("\n请在 3 秒内切换到游戏窗口...")
    time.sleep(3)

    # 测试 1: pydirectinput
    print("\n[测试 1] pydirectinput.move(dx=300, dy=0, relative=True)")
    pydirectinput_move(300, 0)
    time.sleep(2)

    print("\n[测试 2] pydirectinput.move(dx=-300, dy=0, relative=True)")
    pydirectinput_move(-300, 0)
    time.sleep(2)

    # 测试 3: 平滑移动
    print("\n[测试 3] smooth_move(dx=300, dy=0, steps=10, duration=0.5)")
    smooth_move(300, 0, steps=10, duration=0.5)
    time.sleep(2)

    print("\n[测试 4] smooth_move(dx=-300, dy=0, steps=10, duration=0.5)")
    smooth_move(-300, 0, steps=10, duration=0.5)
    time.sleep(2)

    # 测试 5: SendInput
    print("\n[测试 5] sendinput_move(dx=300, dy=0)")
    sendinput_move(300, 0)
    time.sleep(2)

    print("\n[测试 6] sendinput_move(dx=-300, dy=0)")
    sendinput_move(-300, 0)
    time.sleep(2)

    # 测试 7: 平滑 SendInput
    print("\n[测试 7] smooth_sendinput_move(dx=300, dy=0, steps=10, duration=0.5)")
    smooth_sendinput_move(300, 0, steps=10, duration=0.5)
    time.sleep(2)

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n请回答以下问题：")
    print("1. 哪个方法让游戏角色转向了？")
    print("2. 哪个方法的转向最自然？")
    print("3. 是否有方法完全无效？")


if __name__ == "__main__":
    test_all_methods()
