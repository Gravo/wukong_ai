#!/usr/bin/env python3
"""
手柄控制测试脚本 - 不依赖模型，直接测试 vgamepad 控制游戏视角
用法：python test_gamepad_manual.py
目标：聚焦游戏窗口后运行，观察视角是否转动
"""
import time
import sys

try:
    import vgamepad
    VIGEM_AVAILABLE = True
except ImportError:
    print("ERROR: vgamepad not installed. Run: copy vgamepad folder to C:\\Python312\\Lib\\site-packages\\")
    sys.exit(1)

def main():
    pad = vgamepad.VX360Gamepad()
    
    # _prime(): 发送非零偏转激活游戏手柄模式
    print("[1] Priming: sending initial deflection...")
    pad.right_joystick_float(0.1, 0.0)
    pad.update()
    time.sleep(0.2)
    
    # 归零
    pad.right_joystick_float(0.0, 0.0)
    pad.update()
    print("[2] Stick released to neutral")
    time.sleep(0.5)
    
    print("\n[3] Testing right stick X-axis sweep...")
    print("    → Watch the game window (focus it now!)")
    print("    → Values: -0.8 (fast left) → 0.0 → +0.8 (fast right)")
    print("    → Press Ctrl+C to stop\n")
    
    test_values = [-0.80, -0.50, -0.25, 0.00, +0.25, +0.50, +0.80]
    
    for val in test_values:
        pad.right_joystick_float(val, 0.0)
        pad.update()
        direction = "LEFT" if val < 0 else ("RIGHT" if val > 0 else "CENTER")
        print(f"    Right stick X = {val:+.2f} ({direction})")
        time.sleep(1.0)
    
    print("\n[4] Continuous oscillation test...")
    print("    → Hold game window focused, watch camera pan")
    for _ in range(6):
        for val in [-0.50, 0.00, +0.50]:
            pad.right_joystick_float(val, 0.0)
            pad.update()
            direction = "LEFT" if val < 0 else ("RIGHT" if val > 0 else "CENTER")
            print(f"    Right stick X = {val:+.2f} ({direction})")
            time.sleep(0.8)
    
    print("\n[5] Testing left stick (movement)...")
    print("    → Left stick Y: +1.0 = forward, -1.0 = backward")
    for val in [0.0, 1.0, 0.0]:
        pad.left_joystick_float(0.0, val)
        pad.update()
        direction = "FORWARD" if val > 0 else ("BACKWARD" if val < 0 else "CENTER")
        print(f"    Left stick Y = {val:+.2f} ({direction})")
        time.sleep(1.0)
    
    # 释放
    pad.right_joystick_float(0.0, 0.0)
    pad.left_joystick_float(0.0, 0.0)
    pad.update()
    print("\n[OK] Test complete. Stick released.")
    print("\nExpected behavior:")
    print("  - Right stick X controls camera rotation (left/right look)")
    print("  - Left stick Y controls character movement (forward/back)")
    print("  - If camera does NOT move: game may need controller mode enabled")
    print("  - If camera moves WRONG direction: negate the sx value")

if __name__ == '__main__':
    main()
