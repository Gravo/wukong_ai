from pynput.keyboard import Listener, Key
import time
import win32gui
import win32con

import time
from pynput.keyboard import Controller, Key
import win32gui

keyboard_controller = Controller()  # 创建键盘控制器



hwnd_target = win32gui.FindWindow(None,"b1  ")
# hwnd_target = win32gui.FindWindow(None,"无标题 - Notepad")
status_record = False  #默认开始监听
should_stop   = False
recorded_keys = []


# 定义一个回调函数，当按键被按下时会调用
def on_press(key):
    global hwnd_target, should_stop, status_record

    active_hwnd = win32gui.GetForegroundWindow()
    if active_hwnd != hwnd_target or status_record == False: #
        return

    curr_time = time.time()
    recorded_keys.append((key, time.time(),'press'))
    # if hasattr(key,'char'): #如果是字母数字等字符
    #     print(f'{key.char} pressed at {curr_time}')
    #
    # else: #特殊字符,shift, ctrl等
    #     print(f'{key} pressed at {curr_time}')




# 定义一个回调函数，当按键被释放时会调用
def on_release(key):
    curr_time = time.time()
    global hwnd_target, should_stop, status_record

    if hasattr(key, 'char'): #如果是字母数字等字符
        if key.char == 'u' and status_record == False:
            status_record = True
            print(f'status_record changed to {status_record}')
        elif key.char == 'u' and status_record == True:
            status_record = False
            print(f'status_record changed to {status_record}')

    active_hwnd = win32gui.GetForegroundWindow()
    if active_hwnd == hwnd_target and status_record == True:
        recorded_keys.append((key, time.time(),'release'))

        # if hasattr(key,'char'): #如果是字母数字等字符
        #     print(f'{key.char} released at {curr_time}')
        # else: #特殊字符,shift, ctrl等
        #     print(f'{key} released at {curr_time}')

        # 如果按下的是Esc键，则停止监听
        if key == Key.esc:
            should_stop = True


def save_recorded_keys(filename):
    print("come to save_recorded_keys")
    with open(filename, 'w') as file:
        for key, timestamp, status in recorded_keys:
            file.write(f'{key}, {timestamp}, {status}\n')


from collections import defaultdict
from pynput.keyboard import Key

#
# def merge_press_release(recorded_keys):
#     """
#     合并按键记录，只保留第一次 press 和最后一次 release 的记录。
#
#     :param recorded_keys: 包含按键、时间戳和动作类型的列表 [(key, timestamp, action), ...]
#     :return: 合并后的按键记录列表 [(key, press_timestamp, release_timestamp), ...]
#     """
#     merged_keys = []
#     key_status = defaultdict(dict)  # 存储按键的状态信息
#
#     for key, timestamp, action in recorded_keys:
#         if action == 'press':
#             if key not in key_status or key_status[key]['status'] != 'pressed':
#                 key_status[key] = {'status': 'pressed', 'press_timestamp': timestamp}
#         elif action == 'release':
#             if key in key_status and key_status[key]['status'] == 'pressed':
#                 key_status[key]['status'] = 'released'
#                 key_status[key]['release_timestamp'] = timestamp
#                 merged_keys.append((key, key_status[key]['press_timestamp'], key_status[key]['release_timestamp']))
#
#     with open("opt_recorded_keys.txt", 'w') as file:
#         for key, sleep_duration, current_status in merged_keys:
#             file.write(f'{key}, {sleep_duration}, {current_status}\n')
#
#     return merged_keys


from collections import defaultdict
from pynput.keyboard import Key


def merge_press_release(recorded_keys):
    """
    合并按键记录，只保留第一次 press 和最后一次 release 的记录，并保持指定格式。

    :param recorded_keys: 包含按键、时间戳和动作类型的列表 [(key, timestamp, action), ...]
    :return: 合并后的按键记录列表 [(key, press_timestamp, press), ...] 或 [(key, release_timestamp, release), ...]
    """
    merged_keys = []
    key_status = defaultdict(dict)  # 存储按键的状态信息

    for key, timestamp, action in recorded_keys:
        if action == 'press':
            if key not in key_status or key_status[key]['status'] != 'pressed':
                key_status[key] = {'status': 'pressed', 'press_timestamp': timestamp}
        elif action == 'release':
            if key in key_status and key_status[key]['status'] == 'pressed':
                key_status[key]['status'] = 'released'
                key_status[key]['release_timestamp'] = timestamp
                merged_keys.append((key, key_status[key]['press_timestamp'], 'press'))
                merged_keys.append((key, key_status[key]['release_timestamp'], 'release'))

    with open("opt_recorded_keys.txt", 'w') as file:
        for key, sleep_duration, current_status in merged_keys:
            file.write(f'{key}, {sleep_duration}, {current_status}\n')

    return merged_keys




#
# def save_optimize_keys(filename, min_interval=0.01):
#     """
#     对按键记录进行优化，将时间间隔小于 min_interval 的按键合并成一组。
#
#     :param recorded_keys: 包含按键和时间戳的列表 [(key, timestamp), ...]
#     :param min_interval: 最小时间间隔（秒）
#     :return: 优化后的按键记录列表 [(key, timestamp, sleep_duration), ...]
#     """
#     optimized_keys = []
#     current_key = None
#     current_time = None
#     accumulated_keys = []
#
#     for key, timestamp, status in recorded_keys:
#         if current_key is None:
#             current_key = key
#             current_time = timestamp
#             current_status = status
#             accumulated_keys.append((key, min_interval, current_status))
#             continue
#
#         interval = timestamp - current_time
#
#         if key == current_key:
#             if interval < min_interval:
#                 continue
#             else:
#
#             # 如果间隔小于 min_interval 且按键相同，则累加按键
#             # accumulated_keys.append(key, interval, current_status)
#             print(f"opted, {interval}, {current_time}, {timestamp}")
#         else:
#             # 否则，记录当前按键组并开始新的按键组
#             sleep_duration = max(min_interval, timestamp - current_time)
#             optimized_keys.append((current_key,  sleep_duration, current_status))
#
#             current_key = key
#             current_time = timestamp
#             accumulated_keys = [key]
#
#     # 添加最后一个按键组
#     if accumulated_keys:
#         sleep_duration = max(min_interval, timestamp - current_time)
#         optimized_keys.append((current_key,  sleep_duration, current_status))
#
#     with open(filename, 'w') as file:
#         for key, sleep_duration, current_status in accumulated_keys:
#             file.write(f'{key}, {sleep_duration}, {current_status}\n')
#
#     return optimized_keys


def replay_keys(keys):
    start_time = keys[0][1]  # 第一个按键的时间戳作为基准时间
    previous_time = start_time  # 上一个按键的时间戳

    for key, timestamp in keys:
        # 计算当前按键与上一个按键之间的时间间隔
        sleep_duration = timestamp - previous_time

        # 模拟按键按下之前的延迟
        time.sleep(sleep_duration - (time.time() - start_time))

        # # 模拟按键按下
        # print(f'Replaying {key} at {time.time()}')

        # 更新上一个按键的时间戳
        previous_time = timestamp


def replay_keys_from_file(filename):
    start_time = None
    previous_time = None
    global hwnd_target

    with open(filename, 'r') as file:
        lines = file.readlines()

        for line in lines:
            key_str, timestamp_str, key_status = line.strip().split(', ')
            timestamp = float(timestamp_str)

            if start_time is None:
                start_time = timestamp
                previous_time = timestamp
                continue

            sleep_duration = max(0, timestamp - previous_time)  # 确保时间间隔非负
            time.sleep(max(0.001, sleep_duration))  # 设置最小睡眠时间为 0.001 秒

            key = eval(key_str)  # 将字符串转换为 Key 对象

            win32gui.SetForegroundWindow(hwnd_target)  # 将目标窗口置于前台

            # print(key_str, timestamp_str, key_status)
            if key_status == 'press':
                keyboard_controller.press(key)
            else:
                keyboard_controller.release(key)

            previous_time = timestamp


def record_process():
    # 创建一个监听器对象，设置on_press和on_release的回调函数
    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()
    # # 进入主循环，直到监听器停止
    # listener.join()
    while not should_stop:
        time.sleep(0.01)
        pass

    # 停止监听
    listener.stop()
    # print("come to stop")
    # 保存记录的按键到文件
    save_recorded_keys('recorded_keys.txt')
    # merge_press_release(recorded_keys)


if __name__ == '__main__':

    record_process()


    # 从文件读取按键并回放
    # replay_keys_from_file('recorded_keys.txt')


