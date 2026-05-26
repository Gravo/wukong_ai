"""
Test if game responds to different input methods.
Run this WHILE GAME IS FOCUSED.
"""
import time
import ctypes
from ctypes import wintypes

# Method 1: SendInput (what mouse_util.py uses)
def test_sendinput():
    print("\n[Test 1] SendInput - Moving right 500px...")
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
    
    inp = INPUT()
    inp.type = 0  # INPUT_MOUSE
    inp.mi.dx = 500
    inp.mi.dy = 0
    inp.mi.dwFlags = 0x0001  # MOUSEEVENTF_MOVE
    
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    print("  Sent! Did game turn?")


# Method 2: WM_MOUSEMOVE message (send to game window)
def test_wm_mousemove():
    print("\n[Test 2] WM_MOUSEMOVE message...")
    user32 = ctypes.windll.user32
    
    # Find game window
    def enum_callback(hwnd, results):
        title = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, title, 256)
        if 'Black Myth' in title.value or 'Wukong' in title.value or '黑神话' in title.value:
            results.append(hwnd)
        return True
    
    results = []
    EnumWindows = user32.EnumWindows
    EnumWindows(ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(enum_callback), ctypes.byref(ctypes.c_long(0)))
    
    if not results:
        print("  Game window NOT found!")
        return
    
    hwnd = results[0]
    print(f"  Found game window: {hwnd}")
    
    # Send WM_MOUSEMOVE
    WM_MOUSEMOVE = 0x0200
    x, y = 500, 500  # Center-ish
    lParam = (y << 16) | x
    user32.SendMessageW(hwnd, WM_MOUSEMOVE, 0, lParam)
    print("  Sent! Did game turn?")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing input methods for Black Myth: Wukong")
    print("1. Make sure game is running and focused")
    print("2. You have 5 seconds to focus the game")
    print("=" * 60)
    time.sleep(5)
    
    # Test 1: SendInput
    test_sendinput()
    time.sleep(2)
    
    # Test 2: WM_MOUSEMOVE
    test_wm_mousemove()
    time.sleep(2)
    
    print("\n" + "=" * 60)
    print("Results:")
    print("  If Test 1 worked: SendInput is OK, game uses Windows cursor")
    print("  If Test 2 worked: Need to send messages to window")
    print("  If neither worked: Game uses Raw Input (need to disable in settings)")
    print("=" * 60)
