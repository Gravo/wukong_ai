"""test_mouse_buckets.py - 测试不同 bucket 速度的转向效果

手动测试脚本：
  1. 按 'w' 前进
  2. 依次测试每个 bucket (0~6)
  3. 每个 bucket 测试 5 秒
  4. 观察游戏角色是否转向

使用方法：
  1. 打开黑神话悟空游戏
  2. 管理员 PowerShell 运行：
     cd D:\projects\wukong_ai
     C:\Python\python.exe -u test_mouse_buckets.py
  3. 5秒倒计时内聚焦游戏窗口
  4. 观察每个 bucket 的转向效果

Bucket 定义：
  0: 快速左转 (-800 pps)
  1: 中速左转 (-400 pps)
  2: 慢速左转 (-150 pps)
  3: 直行 (0 pps)
  4: 慢速右转 (+150 pps)
  5: 中速右转 (+400 pps)
  6: 快速右转 (+800 pps)
"""

import time
import ctypes
import threading
import pydirectinput
import pyautogui

# ============ 配置 ============
BUCKET_SPEED_PPS = {
    0: -800,  # 快速左转
    1: -400,  # 中速左转
    2: -150,  # 慢速左转
    3: 0,     # 直行
    4: 150,   # 慢速右转
    5: 400,   # 中速右转
    6: 800,   # 快速右转
}

TEST_DURATION_PER_BUCKET = 5  # 每个 bucket 测试 5 秒
MOVE_STEP = 1                   # 每次移动 1 像素
MOVE_INTERVAL = 0.002           # 2ms = 500Hz
STOP_FLAG = False

# ============ 高频移动线程 ============
def _high_freq_move(speed_pps, duration):
    """高频微步移动线程"""
    global STOP_FLAG
    
    if speed_pps == 0 or STOP_FLAG:
        return
    
    direction = 1 if speed_pps > 0 else -1
    steps = int(duration / MOVE_INTERVAL)
    dx = direction * MOVE_STEP
    
    pydirectinput.PAUSE = 0
    pydirectinput.FAILSAFE = True
    
    for _ in range(steps):
        if STOP_FLAG:
            break
        try:
            pydirectinput.moveRel(dx, 0, relative=True)
        except Exception:
            break
        time.sleep(MOVE_INTERVAL)


def test_bucket(bucket_id, speed_pps):
    """测试单个 bucket"""
    global STOP_FLAG
    STOP_FLAG = False
    
    print(f"\n{'='*60}")
    print(f"[测试] Bucket {bucket_id}: {speed_pps:+} pps")
    print(f"{'='*60}")
    
    if speed_pps == 0:
        print(f"[测试] 直行（不移动鼠标）")
        print(f"[测试] 按 Ctrl+C 跳过本次测试...")
        time.sleep(TEST_DURATION_PER_BUCKET)
        return
    
    print(f"[测试] 开始转向（速度={speed_pps:+} pps）")
    print(f"[测试] 按 Ctrl+C 停止...")
    
    # 启动高频移动线程
    t = threading.Thread(target=_high_freq_move, args=(speed_pps, TEST_DURATION_PER_BUCKET))
    t.daemon = True
    t.start()
    
    # 等待测试完成
    try:
        for i in range(TEST_DURATION_PER_BUCKET, 0, -1):
            print(f"  {i}s remaining...", end='\r')
            time.sleep(1)
        print(f"  Done!{' '*20}")
    except KeyboardInterrupt:
        STOP_FLAG = True
        print(f"\n[测试] 停止!")
    
    STOP_FLAG = False
    time.sleep(0.5)  # 等待线程结束


def focus_game_window():
    """聚焦游戏窗口"""
    try:
        def enum_callback(hwnd, results):
            title = ctypes.create_unicode_buffer(256)
            ctypes.windll.user32.GetWindowTextW(hwnd, title, 256)
            if 'Black Myth' in (title.value or '') or '黑神话' in (title.value or '') or 'Wukong' in (title.value or ''):
                results.append(hwnd)
            return True
        
        results = []
        EnumWindows = ctypes.windll.user32.EnumWindows
        EnumWindows(ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)(enum_callback), ctypes.byref(ctypes.c_long(0)))
        
        if results:
            hwnd = results[0]
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            ctypes.windll.user32.ShowWindow(hwnd, 9)
            return True
    except Exception as e:
        print(f"[警告] 聚焦游戏窗口失败: {e}")
    return False


def main():
    global STOP_FLAG
    
    print("=" * 60)
    print("鼠标 Bucket 测试脚本")
    print("=" * 60)
    print("\n[准备] 需要管理员权限运行！")
    print("[准备] 确保黑神话悟空游戏已打开\n")
    
    # 聚焦游戏窗口
    if focus_game_window():
        print("[准备] 游戏窗口已聚焦!")
    else:
        print("[准备] ⚠️  未找到游戏窗口，确保游戏正在运行!")
    
    # 初始化 pydirectinput
    pydirectinput.PAUSE = 0
    pydirectinput.FAILSAFE = True
    pyautogui.PAUSE = 0
    
    # 倒计时
    print("\n[准备] 5秒后开始...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("\n[开始] 测试开始!\n")
    
    try:
        # 按下 W 键（一直前进）
        print("[开始] 按住 W 键（前进）...")
        pyautogui.keyDown('w')
        
        # 测试每个 bucket
        for bucket_id in range(7):
            speed_pps = BUCKET_SPEED_PPS[bucket_id]
            test_bucket(bucket_id, speed_pps)
        
        print("\n" + "=" * 60)
        print("[完成] 所有 bucket 测试完成!")
        print("=" * 60)
        
        # 停止 W 键
        pyautogui.keyUp('w')
        
        # 询问是否需要重新测试
        print("\n[询问] 是否需要重新测试某个 bucket？")
        print("[询问] 输入 bucket ID (0-6)，或输入 'q' 退出")
        
        while True:
            try:
                user_input = input("[询问] 你的选择: ").strip()
                if user_input.lower() == 'q':
                    break
                
                bucket_id = int(user_input)
                if 0 <= bucket_id <= 6:
                    speed_pps = BUCKET_SPEED_PPS[bucket_id]
                    print(f"\n[重新测试] Bucket {bucket_id}: {speed_pps:+} pps")
                    pyautogui.keyDown('w')
                    test_bucket(bucket_id, speed_pps)
                    pyautogui.keyUp('w')
                else:
                    print("[错误] 请输入 0-6 之间的数字")
            except ValueError:
                print("[错误] 请输入数字或 'q'")
        
    except KeyboardInterrupt:
        print("\n\n[停止] 用户中断!")
    
    finally:
        STOP_FLAG = True
        time.sleep(0.1)
        pyautogui.keyUp('w')
        print("\n[清理] 已释放所有按键")


if __name__ == "__main__":
    main()
