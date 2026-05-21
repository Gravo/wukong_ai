"""
L2 辅助驾驶数据采集器

专门用于采集辅助功能训练数据：
1. 闪避数据：攻击前摇 + 闪避动作
2. 面敌数据：敌人位置 + 鼠标修正
3. 连招数据：敌人状态 + 攻击序列

使用方式：
    # 采集闪避数据
    python assist/data_collector.py --mode dodge --duration 300

    # 采集面敌数据
    python assist/data_collector.py --mode face --duration 300

    # 采集所有数据
    python assist/data_collector.py --mode all --duration 300
"""
import os
import sys
import time
import argparse
import numpy as np
import cv2
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GAME_REGION, NUM_ACTIONS
from env.screen_capture import ScreenCapture


class L2DataCollector:
    """
    L2 辅助功能数据采集器

    采集模式：
    - dodge: 闪避数据（攻击前摇 + 闪避动作）
    - face: 面敌数据（敌人位置 + 鼠标修正）
    - all: 所有数据
    """

    def __init__(self, output_dir: str = "l2_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.capture = ScreenCapture()

        # 键盘状态
        self._keys_pressed = set()
        self._recording = False

        # 鼠标状态
        self._mouse_dx = 0
        self._mouse_dy = 0
        self._last_mouse_pos = None

        # 数据缓冲区
        self.dodge_data: List[Dict[str, Any]] = []
        self.face_data: List[Dict[str, Any]] = []

    def setup_keyboard(self):
        """设置键盘监听"""
        import keyboard as kb

        try:
            kb.unhook_all()
        except Exception:
            pass

        MONITORED = {'w', 'a', 's', 'd', 'j', 'k', 'v', 'r', 'space', 'shift'}

        def on_key(e):
            name = e.name.lower()
            if name in MONITORED:
                if e.event_type == kb.KEY_DOWN:
                    self._keys_pressed.add(name)
                elif e.event_type == kb.KEY_UP:
                    self._keys_pressed.discard(name)
            if name == 'esc' and e.event_type == kb.KEY_DOWN:
                self._recording = False

        kb.hook(on_key)
        return True

    def setup_mouse(self):
        """设置鼠标监听"""
        from pynput import mouse

        def on_move(x, y):
            if self._last_mouse_pos is None:
                self._last_mouse_pos = (x, y)
                return
            dx = x - self._last_mouse_pos[0]
            dy = y - self._last_mouse_pos[1]
            self._last_mouse_pos = (x, y)
            self._mouse_dx += dx
            self._mouse_dy += dy

        self._mouse_listener = mouse.Listener(on_move=on_move)
        self._mouse_listener.start()
        return True

    def get_mouse_state(self):
        """获取鼠标状态并重置"""
        dx = self._mouse_dx
        dy = self._mouse_dy
        self._mouse_dx = 0
        self._mouse_dy = 0
        return dx, dy

    def is_dodging(self) -> bool:
        """检测玩家是否在闪避"""
        return 'space' in self._keys_pressed

    def is_attacking(self) -> bool:
        """检测玩家是否在攻击"""
        return 'j' in self._keys_pressed

    def collect_dodge_data(self, duration: int = 300, fps: int = 15):
        """
        采集闪避数据

        采集策略：
        1. 录制游戏画面
        2. 标记闪避时刻（玩家按 Space）
        3. 保存连续帧序列

        数据格式：
        - frames: 连续 4 帧画面
        - is_dodge: 是否闪避（0/1）
        """
        print(f"\n{'=' * 60}")
        print(f"  闪避数据采集模式")
        print(f"  持续时间: {duration}秒 | 帧率: {fps}fps")
        print(f"  按 Space 闪避 | 按 ESC 停止")
        print(f"{'=' * 60}\n")

        self.setup_keyboard()
        self.setup_mouse()

        # 等待用户切换到游戏
        print("  切换到游戏窗口... (3秒)")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)

        self._recording = True
        start_time = time.time()
        frame_interval = 1.0 / fps

        # 帧缓冲区（用于堆叠）
        frame_buffer = []
        buffer_size = 4

        samples = []
        sample_count = 0

        print("  开始录制...\n")

        while self._recording:
            loop_start = time.time()
            elapsed = time.time() - start_time

            # 检查时间
            if elapsed >= duration:
                print(f"\n  时间到 ({duration}秒)")
                break

            # 截图
            frame = self.capture.grab()
            if frame is None:
                time.sleep(0.01)
                continue

            # 缩放到 224x224
            small = cv2.resize(frame, (224, 224))

            # 添加到缓冲区
            frame_buffer.append(small)
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)

            # 如果缓冲区满，保存样本
            if len(frame_buffer) == buffer_size:
                is_dodging = 1 if self.is_dodging() else 0

                # 保存帧堆叠
                stacked = np.concatenate([
                    f.transpose(2, 0, 1) for f in frame_buffer
                ], axis=0)

                samples.append({
                    "frames": stacked,
                    "is_dodge": is_dodging,
                    "timestamp": elapsed,
                })

                sample_count += 1

            # 进度显示
            if sample_count % (fps * 5) == 0 and sample_count > 0:
                dodge_count = sum(1 for s in samples if s["is_dodge"] == 1)
                print(f"  [{elapsed:.0f}s] 样本: {sample_count} | 闪避: {dodge_count}")

            # 控制帧率
            dt = time.time() - loop_start
            if dt < frame_interval:
                time.sleep(frame_interval - dt)

        # 保存数据
        if samples:
            self._save_dodge_data(samples)
            print(f"\n  采集完成! 共 {len(samples)} 个样本")

            # 统计
            dodge_count = sum(1 for s in samples if s["is_dodge"] == 1)
            print(f"  闪避样本: {dodge_count} ({dodge_count / len(samples) * 100:.1f}%)")
            print(f"  非闪避样本: {len(samples) - dodge_count}")

        # 清理
        self._cleanup()

    def _save_dodge_data(self, samples: List[Dict[str, Any]]):
        """保存闪避数据"""
        import h5py

        timestamp = int(time.time())
        filepath = os.path.join(self.output_dir, f"dodge_data_{timestamp}.h5")

        frames = np.array([s["frames"] for s in samples], dtype=np.uint8)
        labels = np.array([s["is_dodge"] for s in samples], dtype=np.int8)
        timestamps = np.array([s["timestamp"] for s in samples], dtype=np.float32)

        with h5py.File(filepath, "w") as f:
            f.create_dataset("frames", data=frames, compression="gzip")
            f.create_dataset("labels", data=labels)
            f.create_dataset("timestamps", data=timestamps)
            f.attrs["type"] = "dodge"
            f.attrs["num_samples"] = len(samples)
            f.attrs["fps"] = 15

        sz = os.path.getsize(filepath) / 1024 / 1024
        print(f"  保存到: {filepath} ({sz:.1f}MB)")

    def _cleanup(self):
        """清理资源"""
        import keyboard as kb
        try:
            kb.unhook_all()
        except Exception:
            pass
        if hasattr(self, '_mouse_listener'):
            self._mouse_listener.stop()
        self.capture.release()


def main():
    parser = argparse.ArgumentParser(description="L2 辅助功能数据采集器")
    parser.add_argument("--mode", choices=["dodge", "face", "all"], default="dodge",
                        help="采集模式")
    parser.add_argument("--duration", type=int, default=300, help="采集时长（秒）")
    parser.add_argument("--fps", type=int, default=15, help="帧率")
    parser.add_argument("--output", default="l2_data", help="输出目录")

    args = parser.parse_args()

    collector = L2DataCollector(output_dir=args.output)

    if args.mode == "dodge":
        collector.collect_dodge_data(duration=args.duration, fps=args.fps)
    elif args.mode == "face":
        print("[TODO] 面敌数据采集模式待实现")
    elif args.mode == "all":
        print("[TODO] 全量数据采集模式待实现")


if __name__ == "__main__":
    main()
