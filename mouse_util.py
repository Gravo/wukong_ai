"""
Windows SendInput mouse mover - works with any game (Raw Input / DirectInput / Win32).
Save this file to: D:\projects\wukong_ai\mouse_util.py
"""
import ctypes
from ctypes import wintypes
import time

# Windows API constants
MOUSEEVENTF_MOVE = 0x0001
INPUT_MOUSE = 0

# Load user32.dll
user32 = ctypes.windll.user32

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

def move(dx, dy):
    """
    Move mouse RELATIVE to current position.
    Works with any game (DirectInput, Raw Input, etc.)
    
    Args:
        dx: pixels to move right (negative = left)
        dy: pixels to move down (negative = up)
    """
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.dx = dx
    inp.mi.dy = dy
    inp.mi.mouseData = 0
    inp.mi.dwFlags = MOUSEEVENTF_MOVE  # Relative movement
    inp.mi.time = 0
    inp.mi.dwExtraInfo = None
    
    # Send the input
    result = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if result != 1:
        error_code = ctypes.GetLastError()
        raise WindowsError(f"SendInput failed: error code {error_code}")

def move_smooth(dx, dy, steps=5):
    """
    Smooth mouse movement (more human-like).
    
    Args:
        dx: total pixels to move right
        dy: total pixels to move down
        steps: number of small movements
    """
    step_x = dx // steps
    step_y = dy // steps
    for _ in range(steps):
        move(step_x, step_y)
        time.sleep(0.01)  # 10ms between steps

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SendInput mouse move...")
    print("1. Make sure the game is running")
    print("2. Focus the game window in 3 seconds!")
    print("=" * 60)
    time.sleep(3)
    
    print("\n[Test 1] Moving right 300px...")
    move(300, 0)
    time.sleep(1)
    
    print("[Test 2] Moving left 300px...")
    move(-300, 0)
    time.sleep(1)
    
    print("\n[Test 3] Moving right 120px (turn_slow)...")
    move(120, 0)
    time.sleep(0.5)
    
    print("[Test 4] Moving right 300px (turn_medium)...")
    move(300, 0)
    time.sleep(0.5)
    
    print("\nDone! Did the character turn?")
    print("If YES: SendInput works!")
    print("If NO: Game might have anti-cheat or special input handling.")
