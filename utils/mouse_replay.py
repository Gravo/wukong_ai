from pynput import mouse
import time

# 创建一个列表来保存鼠标移动的位置
mouse_movements = []

# 定义一个回调函数，当鼠标移动时被调用
def on_move(x, y):
    # 记录当前时间和位置
    timestamp = time.time()
    print(f"Mouse moved to ({x}, {y}) at {timestamp}")
    mouse_movements.append((timestamp, x, y))


def save_mouse_movement():
    # 设置鼠标监听器
    with mouse.Listener(on_move=on_move) as listener:
        try:
            # 保持程序运行，直到手动停止
            listener.join()
        except KeyboardInterrupt:
            # 当用户按下Ctrl+C时，优雅地退出
            print("Recording stopped.")
            pass
    # 将记录的数据写入文件
    with open('mouse_movements.txt', 'w') as f:
        for movement in mouse_movements:
            f.write(f"{movement[0]} {movement[1]} {movement[2]}\n")


save_mouse_movement()
def replay_mouse_movement():
    # 读取记录文件
    with open('mouse_movements.txt', 'r') as f:
        lines = f.readlines()
    # 解析每一行数据
    movements = []
    for line in lines:
        timestamp, x, y = map(float, line.strip().split())
        movements.append((timestamp, int(x), int(y)))
    # 按照时间顺序排序（如果需要的话）
    movements.sort(key=lambda m: m[0])
    # 回放鼠标移动
    start_time = None
    for i, (timestamp, x, y) in enumerate(movements):
        if i == 0:
            # 记录开始时间
            start_time = time.time()

        # 计算应该等待的时间
        wait_time = timestamp - (start_time + (time.time() - start_time))
        if wait_time > 0:
            time.sleep(wait_time)

        # 移动鼠标
        pyautogui.moveTo(x, y)

        # 如果你需要模拟点击，可以在这里添加逻辑
        # 例如，假设你的文件中还包含了点击信息，可以这样做：
        # if click_info:
        #     pyautogui.click()
    print("Mouse movements replayed.")


# replay_mouse_movement()