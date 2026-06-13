#!/usr/bin/env python3
"""验证 vgamepad 安装状态"""
import sys

print("=" * 50)
print("vgamepad 安装验证")
print("=" * 50)

try:
    import vgamepad
    print("[OK] vgamepad 导入成功")
    print(f"     版本: {getattr(vgamepad, '__version__', 'unknown')}")
    print(f"     模块路径: {vgamepad.__file__}")
except ImportError as e:
    print(f"[FAIL] vgamepad 导入失败: {e}")
    print("  解决方案：")
    print("  1. 下载 vgamepad-0.1.0.tar.gz")
    print("  2. 解压到 site-packages 目录")
    print("  3. 或运行: pip install vgamepad")
    sys.exit(1)

# 检查 DLL
import os
dll_path = os.path.join(os.path.dirname(vgamepad.__file__), 'win', 'vigem', 'client', 'x64', 'ViGEmClient.dll')
if os.path.exists(dll_path):
    print(f"[OK] ViGEmClient.dll 存在: {dll_path}")
else:
    print(f"[WARN] ViGEmClient.dll 未找到: {dll_path}")
    print("  vgamepad 仍可尝试运行（可能在其他位置）")

# 列出可用类
classes = [k for k in dir(vgamepad) if not k.startswith('_')]
print(f"[OK] 可用类/函数: {', '.join(classes)}")

# 检查 VX360Gamepad
if hasattr(vgamepad, 'VX360Gamepad'):
    print("[OK] VX360Gamepad 类可用")
    print("\n尝试创建虚拟手柄...")
    print("  (如果 ViGEmBus 驱动未安装，将报错)")
    try:
        pad = vgamepad.VX360Gamepad()
        print("[OK] 虚拟 Xbox 360 手柄创建成功！")
        # 测试基本调用
        pad.right_joystick_float(0.5, 0.0)
        pad.update()
        pad.right_joystick_float(0.0, 0.0)
        pad.update()
        del pad
        print("[OK] 手柄操作测试通过")
        print("\n[SUCCESS] vgamepad 安装验证完成！")
        print("  可以运行 test_gamepad_manual.py 测试游戏内视角控制")
    except Exception as e:
        print(f"[FAIL] 手柄创建失败: {e}")
        print("  可能原因：ViGEmBus 驱动未安装或未启动")
        print("  解决方案：")
        print("  1. 安装 ViGEmBus: https://github.com/nefarius/ViGEmBus/releases")
        print("  2. 或者 vgamepad 内置的 MSI: C:\\Python312\\Lib\\site-packages\\vgamepad\\win\\vigem\\install\\x64\\ViGEmBusSetup_x64.msi")
        sys.exit(1)
else:
    print("[FAIL] VX360Gamepad 类不可用")
    sys.exit(1)
