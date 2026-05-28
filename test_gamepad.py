"""
test_gamepad.py - 测试 ViGEmBus 虚拟 Xbox 360 手柄控制黑神话悟空

原理：
- 黑神话支持手柄，右摇杆 = 视角控制
- ViGEmBus 创建虚拟 Xbox 360 手柄（内核级 HID 设备）
- 游戏无法区分虚拟手柄和真实手柄
- 右摇杆映射到视角转动，完全绕过鼠标 Raw Input 问题

前置条件：
1. 安装 ViGEmBus 驱动：https://github.com/nefarius/ViGEmBus/releases/tag/v1.22.0
2. 安装 Python 库：pip install vigemclient

测试步骤：
1. 运行本脚本
2. 切换到黑神话游戏窗口
3. 观察角色是否转动（右摇杆模拟）
4. 按 ESC 退出
"""

import ctypes
import time
import sys

# 尝试导入 vigemclient
try:
    import vigemclient
except ImportError:
    print("=" * 60)
    print("错误：vigemclient 未安装")
    print("=" * 60)
    print("\n安装步骤：")
    print("1. 安装 ViGEmBus 驱动：")
    print("   下载 https://github.com/nefarius/ViGEmBus/releases/tag/v1.22.0")
    print("   运行 ViGEmBusSetup_x64.msi")
    print("   重启电脑")
    print("")
    print("2. 安装 Python 库：")
    print("   C:\\Python\\python.exe -m pip install vigemclient")
    print("")
    print("3. 重新运行本脚本")
    sys.exit(1)


def test_gamepad():
    """测试虚拟 Xbox 360 手柄"""
    print("初始化 ViGEmBus 客户端...")
    client = vigemclient.VigemClient()
    client.connect()
    
    print("创建虚拟 Xbox 360 手柄...")
    gamepad = vigemclient.VigemXbox360Controller(client)
    gamepad.register()
    
    print("虚拟手柄已创建！")
    print("5 秒后开始测试，请切换到游戏窗口...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    # Xbox 360 右摇杆值范围：-32768 到 32767
    # 右摇杆 X 轴 = 视角水平转动
    # 右摇杆 Y 轴 = 视角垂直转动
    
    print("\n--- 测试 1：右摇杆向右（视角右转）---")
    gamepad.set_axis_value(vigemclient.Xbox360Axis.RIGHT_THUMB_X, 32767)  # 最大右
    time.sleep(2)
    gamepad.set_axis_value(vigemclient.Xbox360Axis.RIGHT_THUMB_X, 0)  # 回中
    time.sleep(1)
    
    print("--- 测试 2：右摇杆向左（视角左转）---")
    gamepad.set_axis_value(vigemclient.Xbox360Axis.RIGHT_THUMB_X, -32768)  # 最大左
    time.sleep(2)
    gamepad.set_axis_value(vigemclient.Xbox360Axis.RIGHT_THUMB_X, 0)
    time.sleep(1)
    
    print("--- 测试 3：右摇杆慢速右转 ---")
    gamepad.set_axis_value(vigemclient.Xbox360Axis.RIGHT_THUMB_X, 16384)  # 约一半
    time.sleep(3)
    gamepad.set_axis_value(vigemclient.Xbox360Axis.RIGHT_THUMB_X, 0)
    time.sleep(1)
    
    print("--- 测试 4：左摇杆向前移动 ---")
    gamepad.set_axis_value(vigemclient.Xbox360Axis.LEFT_THUMB_Y, -32768)  # 推左摇杆向前
    time.sleep(2)
    gamepad.set_axis_value(vigemclient.Xbox360Axis.LEFT_THUMB_Y, 0)
    time.sleep(1)
    
    print("--- 测试 5：向前+右转（组合）---")
    gamepad.set_axis_value(vigemclient.Xbox360Axis.LEFT_THUMB_Y, -32768)  # 前进
    gamepad.set_axis_value(vigemclient.Xbox360Axis.RIGHT_THUMB_X, 16384)  # 右转
    time.sleep(3)
    gamepad.set_axis_value(vigemclient.Xbox360Axis.LEFT_THUMB_Y, 0)
    gamepad.set_axis_value(vigemclient.Xbox360Axis.RIGHT_THUMB_X, 0)
    time.sleep(1)
    
    print("\n测试完成！请观察游戏内是否有响应。")
    print("按 Enter 退出...")
    input()
    
    gamepad.unregister()
    client.disconnect()
    print("虚拟手柄已断开。")


if __name__ == "__main__":
    test_gamepad()
