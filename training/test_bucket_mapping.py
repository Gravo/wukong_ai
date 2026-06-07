#!C:\Python\python.exe
#!/usr/bin/env python3
"""测试脚本1：Bucket → 右摇杆 映射详细测试"""
import time
import vgamepad
import sys

BUCKET_STICK = {0: -0.80, 1: -0.50, 2: -0.25, 3: 0.00, 4: +0.25, 5: +0.50, 6: +0.80}
BUCKET_NAME   = {
    0: "快速左转",
    1: "中速左转",
    2: "慢速左转",
    3: "直行(不转)",
    4: "慢速右转",
    5: "中速右转",
    6: "快速右转",
}

print("=" * 60)
print("Bucket → 右摇杆 映射测试")
print("请先聚焦《黑神话：悟空》游戏窗口！")
print("每个 bucket 停留 2 秒，观察视角转动方向")
print("=" * 60)

pad = vgamepad.VX360Gamepad()

# prime：发送非零偏转，激活游戏手柄模式
pad.right_joystick_float(0.1, 0.0)
pad.update()
time.sleep(0.2)
pad.right_joystick_float(0.0, 0.0)
pad.left_joystick_float(0.0, 0.0)
pad.update()
print("[OK] vgamepad initialized\n")

try:
    for bucket in range(7):
        sx = BUCKET_STICK[bucket]
        name = BUCKET_NAME[bucket]

        # 右摇杆：控制视角（X轴）
        pad.right_joystick_float(float(sx), 0.0)
        # 左摇杆：保持前进（Y轴=1.0），方便观察
        pad.left_joystick_float(0.0, 1.0)
        pad.update()

        direction = "← 左转" if sx < 0 else ("静止" if sx == 0.0 else "右转 →")
        print(f"  bucket={bucket}  {name:12s}  sx={sx:+.2f}  {direction}")

        time.sleep(2.0)

    # 回到中心
    pad.right_joystick_float(0.0, 0.0)
    pad.left_joystick_float(0.0, 0.0)
    pad.update()

    print("\n" + "=" * 60)
    print("测试完成！请观察：")
    print("  1. 视角是否转动？")
    print("     → 如果完全不动：游戏可能不支持右摇杆控制视角")
    print("     → 如果只有部分转动：max_deflection 可能需要调大")
    print("  2. 方向对不对？")
    print("     → 如果 bucket=0/1/2 反而右转：需要把 BUCKET_STICK 符号取反")
    print("  3. 灵敏度合适吗？")
    print("     → 转动太快：把 -0.80~+0.80 改小，比如 -0.50~+0.50")
    print("     → 转动太慢：把 -0.80~+0.80 改大，最大 ±1.0")
    print("=" * 60)

except KeyboardInterrupt:
    pad.right_joystick_float(0.0, 0.0)
    pad.left_joystick_float(0.0, 0.0)
    pad.update()
    print("\n[中断] 已释放摇杆")

except Exception as e:
    pad.right_joystick_float(0.0, 0.0)
    pad.left_joystick_float(0.0, 0.0)
    print(f"\n[错误] {e}")
    sys.exit(1)
