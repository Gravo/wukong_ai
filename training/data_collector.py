"""
data_collector.py - 数据采集脚本（增强版）
录制人类玩家的游戏画面+操作，用于行为克隆和训练分析

增强功能：
- 屏幕左上角实时HUD叠加显示录制状态
- 键盘监听诊断（启动时检测按键是否正常捕获）
- F1停止 + 可视化提示
- 实时帧率显示
- 动作名称实时显示
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GAME_REGION, NUM_ACTIONS, ACTION_SPACE
from env.screen_capture import ScreenCapture

# 动作ID -> 中文名称
ACTION_NAMES = {
    0: "IDLE(空闲)",
    1: "ATTACK(攻击)",
    2: "HEAVY(重击)",
    3: "DODGE(闪避)",
    4: "FORWARD(前进)",
    5: "RIGHT(右移)",
    6: "LEFT(左移)",
    7: "DODGE_ATK(闪攻)",
    8: "LOCK(锁定)",
    9: "HEAL(喝药)",
}


class DataCollector:
    """人类游戏数据采集器"""

    def __init__(self, output_dir="pathfinding_data", fps=30, hud=True):
        self.output_dir = output_dir
        self.fps = fps
        self.hud = hud
        self.capture = ScreenCapture()

        os.makedirs(output_dir, exist_ok=True)

        # 按键监听
        self._recording = False
        self._keys_pressed = set()
        self._key_log = []  # 记录最近按键事件，用于诊断

    def setup_keyboard_listener(self):
        """设置键盘监听"""
        from pynput import keyboard

        def on_press(key):
            try:
                char = key.char.lower()
                self._keys_pressed.add(char)
                self._key_log.append(f"DN:{char}")
            except AttributeError:
                name = ""
                if key == keyboard.Key.space:
                    name = "space"
                elif key == keyboard.Key.shift:
                    name = "shift"
                elif key == keyboard.Key.f1:
                    name = "F1"
                if name:
                    self._keys_pressed.add(name)
                    self._key_log.append(f"DN:{name}")

            if key == keyboard.Key.f1:
                self._recording = False
                # Do NOT return False - it kills the listener permanently
                # collect() loop handles _recording flag; test_keyboard() is one-shot

            # 保留最近50条
            if len(self._key_log) > 50:
                self._key_log = self._key_log[-50:]

        def on_release(key):
            try:
                char = key.char.lower()
                self._keys_pressed.discard(char)
                self._key_log.append(f"UP:{char}")
            except AttributeError:
                name = ""
                if key == keyboard.Key.space:
                    name = "space"
                elif key == keyboard.Key.shift:
                    name = "shift"
                elif key == keyboard.Key.f1:
                    name = "F1"
                if name:
                    self._keys_pressed.discard(name)
                    self._key_log.append(f"UP:{name}")

            if len(self._key_log) > 50:
                self._key_log = self._key_log[-50:]

        self._listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
        )
        self._listener.start()

    def test_keyboard(self):
        """测试键盘监听是否正常工作"""
        import cv2

        print("\n" + "=" * 50)
        print("  键盘监听测试")
        print("  请依次按下: W  A  S  D  Space  J  F1(停止测试)")
        print("  如果看到按键被捕获，说明正常")
        print("=" * 50)

        self.setup_keyboard_listener()
        self._recording = True

        start = time.time()
        while self._recording and time.time() - start < 15:
            frame = self.capture.grab()
            if frame is None:
                time.sleep(0.1)
                continue

            # 画测试HUD
            h, w = frame.shape[:2]
            overlay = frame.copy()

            keys_str = ", ".join(sorted(self._keys_pressed)) if self._keys_pressed else "(无)"
            recent = self._key_log[-8:]

            # 半透明背景
            cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            cv2.putText(frame, "[KEYBOARD TEST] Press W/A/S/D/Space/J/F1",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Current keys: {keys_str}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            y = 105
            for log in recent:
                color = (0, 200, 255) if log.startswith("DN") else (100, 100, 100)
                cv2.putText(frame, log, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 22

            cv2.imshow("WukongAI - Key Test", cv2.resize(frame, (640, 360)))
            cv2.waitKey(1)
            time.sleep(0.03)

        cv2.destroyAllWindows()
        captured_count = len([l for l in self._key_log if l.startswith("DN")])

        if captured_count > 0:
            print(f"\n  OK! 捕获了 {captured_count} 次按键，键盘监听正常")
            return True
        else:
            print(f"\n  WARNING: 未捕获到任何按键！")
            print(f"  可能原因：")
            print(f"    1. 以管理员权限运行（某些游戏可能需要）")
            print(f"    2. 杀毒软件拦截了键盘钩子")
            print(f"    3. 游戏反作弊拦截了键盘监听")
            print(f"    请尝试以管理员身份运行此脚本")
            return False

    def keys_to_action_id(self):
        """将当前按键状态映射到动作ID"""
        keys = self._keys_pressed

        # 优先级: 攻击 > 闪避 > 移动 > idle
        if 'j' in keys:
            if 'space' in keys:
                return 7  # dodge_attack
            return 1  # attack

        if 'space' in keys:
            return 3  # dodge

        if 'w' in keys and 'd' in keys:
            return 4  # move_forward (simplified)
        if 'w' in keys:
            return 4  # move_forward
        if 'a' in keys:
            return 6  # move_left
        if 'd' in keys:
            return 5  # move_right

        if 'v' in keys:
            return 8  # lock_on
        if 'r' in keys:
            return 9  # heal

        return 0  # idle

    def draw_hud(self, frame, ep, total_ep, frame_count, fps_actual, action_id, duration, keys_pressed):
        """在帧上绘制HUD叠加层"""
        import cv2

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 半透明黑色背景 (左上角)
        bar_w = 520
        bar_h = 195
        cv2.rectangle(overlay, (10, 10), (10 + bar_w, 10 + bar_h), (0, 0, 0), -1)
        result = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        y = 38
        # 录制状态
        blink = int(time.time() * 2) % 2 == 0  # 闪烁
        rec_color = (0, 0, 255) if blink else (0, 0, 180)
        cv2.circle(result, (25, y - 5), 8, rec_color, -1)
        cv2.putText(result, "REC", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, rec_color, 2)

        # Episode
        cv2.putText(result, f"Episode: {ep}/{total_ep}",
                     (110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # F1提示
        cv2.putText(result, "[F1] STOP",
                     (330, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        y += 32
        # 帧数和时长
        cv2.putText(result, f"Frames: {frame_count}  FPS: {fps_actual:.1f}  Time: {duration:.1f}s",
                     (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        y += 28
        # 当前动作
        action_name = ACTION_NAMES.get(action_id, f"UNK({action_id})")
        if action_id == 0:
            action_color = (100, 100, 100)
        elif action_id in (1, 2, 7):
            action_color = (0, 100, 255)
        elif action_id == 3:
            action_color = (0, 255, 255)
        else:
            action_color = (0, 255, 0)

        cv2.putText(result, f"Action: {action_name}",
                     (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, action_color, 2)

        y += 28
        # 当前按键
        keys_str = ", ".join(sorted(keys_pressed)) if keys_pressed else "(none)"
        cv2.putText(result, f"Keys: {keys_str}",
                     (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # 动作分布小条 (右下角)
        if not hasattr(self, '_action_counts'):
            self._action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        self._action_counts[action_id] += 1
        total = max(self._action_counts.sum(), 1)

        bar_x = w - 180
        bar_y = h - 220
        cv2.rectangle(result, (bar_x - 10, bar_y - 25), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.putText(result, "Actions", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        bar_y += 5
        for i in range(NUM_ACTIONS):
            count = self._action_counts[i]
            if count > 0:
                pct = count / total
                # 背景条
                cv2.rectangle(result, (bar_x, bar_y), (bar_x + 150, bar_y + 14), (50, 50, 50), -1)
                # 填充条
                bar_fill = int(150 * pct)
                colors = [(100,100,100),(0,100,255),(0,80,200),(0,255,255),(0,200,0),(0,200,0),(0,200,0),(255,100,0),(200,200,0),(0,255,0)]
                cv2.rectangle(result, (bar_x, bar_y), (bar_x + bar_fill, bar_y + 14), colors[i], -1)
                # 标签
                short_names = ["idle","atk","heavy","dodge","fwd","rgt","lft","d_atk","lock","heal"]
                cv2.putText(result, f"{short_names[i]}:{count}",
                             (bar_x + 3, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
                bar_y += 17

        return result

    def collect(self, num_episodes=5, mode="pathfinding", skip_test=False):
        """
        录制数据

        Args:
            num_episodes: 录制几个episode
            mode: "pathfinding" | "combat"
            skip_test: 跳过键盘测试
        """
        import h5py
        import cv2

        print(f"\n{'=' * 60}")
        print(f"  WukongAI Data Collector (Enhanced)")
        print(f"  Mode: {mode}  |  Episodes: {num_episodes}  |  FPS: {self.fps}")
        print(f"  Output: {os.path.abspath(self.output_dir)}")
        print(f"{'=' * 60}")

        # Step 1: 键盘测试
        if not skip_test:
            print()
            ok = self.test_keyboard()
            if not ok:
                print("\n  键盘监听异常！建议以管理员权限运行。")
                choice = input("  是否继续? (y/n): ").strip().lower()
                if choice != 'y':
                    print("  已退出。")
                    return
            # 重建键盘监听（测试时可能被消耗）
            self._keys_pressed = set()
            self._key_log = []
            self._listener.stop()
            self.setup_keyboard_listener()

        print(f"\n{'=' * 60}")
        print(f"  开始正式录制")
        print(f"  切换到游戏窗口... (3秒倒计时)")
        print(f"  按 F1 停止当前episode")
        print(f"{'=' * 60}")

        # 倒计时
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)

        all_results = []

        for ep in range(num_episodes):
            episode_frames = []
            episode_actions = []
            episode_timestamps = []

            self._action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
            self._recording = True

            print(f"\n  [Episode {ep+1}/{num_episodes}] RECORDING... (F1 to stop)")

            start_time = time.time()
            frame_interval = 1.0 / self.fps
            fps_samples = []

            while self._recording:
                loop_start = time.time()

                # 截图
                frame = self.capture.grab()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # 获取动作
                action_id = self.keys_to_action_id()

                # 绘制HUD（在原始分辨率上）
                duration = time.time() - start_time
                fps_actual = len(episode_frames) / max(duration, 0.001)

                if self.hud:
                    frame = self.draw_hud(
                        frame, ep + 1, num_episodes,
                        len(episode_frames), fps_actual,
                        action_id, duration, self._keys_pressed
                    )
                    # 显示预览窗口
                    preview = cv2.resize(frame, (640, 360))
                    cv2.imshow("WukongAI - Recording", preview)
                    cv2.waitKey(1)

                # 缩小帧保存
                small_frame = cv2.resize(frame, (224, 224))
                episode_frames.append(small_frame)
                episode_actions.append(action_id)
                episode_timestamps.append(time.time() - start_time)

                # 控制帧率
                elapsed = time.time() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

            # 停止后关闭预览
            if self.hud:
                cv2.destroyAllWindows()

            # 保存episode数据
            if len(episode_frames) > 10:
                save_path = os.path.join(
                    self.output_dir,
                    f"{mode}_ep{ep+1}_{int(time.time())}.h5"
                )

                with h5py.File(save_path, "w") as f:
                    f.create_dataset(
                        "frames",
                        data=np.array(episode_frames, dtype=np.uint8),
                        compression="gzip",
                    )
                    f.create_dataset(
                        "actions",
                        data=np.array(episode_actions, dtype=np.int8),
                    )
                    f.create_dataset(
                        "timestamps",
                        data=np.array(episode_timestamps, dtype=np.float32),
                    )
                    f.attrs["mode"] = mode
                    f.attrs["fps"] = self.fps
                    f.attrs["num_frames"] = len(episode_frames)

                # 统计
                action_counts = np.bincount(
                    np.array(episode_actions),
                    minlength=NUM_ACTIONS,
                )
                result = {
                    "path": save_path,
                    "frames": len(episode_frames),
                    "duration": episode_timestamps[-1],
                    "fps_actual": len(episode_frames) / episode_timestamps[-1],
                    "actions": {ACTION_SPACE[i][0]: int(action_counts[i]) for i in range(NUM_ACTIONS) if action_counts[i] > 0},
                }
                all_results.append(result)

                print(f"\n  SAVED: {os.path.basename(save_path)}")
                print(f"  Frames: {len(episode_frames)}  Duration: {episode_timestamps[-1]:.1f}s  FPS: {result['fps_actual']:.1f}")
                for name, count in result["actions"].items():
                    print(f"    {name}: {count} ({count/len(episode_frames)*100:.1f}%)")
            else:
                print(f"  Too few frames ({len(episode_frames)}), skipped")

            if ep < num_episodes - 1:
                print(f"\n  Next episode in 3s...")
                time.sleep(3)

        self._listener.stop()
        self.capture.release()

        # 总结
        print(f"\n{'=' * 60}")
        print(f"  DONE! {len(all_results)} episodes saved to:")
        print(f"  {os.path.abspath(self.output_dir)}")
        total_frames = sum(r["frames"] for r in all_results)
        total_time = sum(r["duration"] for r in all_results)
        print(f"  Total: {total_frames} frames, {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"{'=' * 60}")

        return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WukongAI - Enhanced Data Collector")
    parser.add_argument("--mode", choices=["pathfinding", "combat"], default="pathfinding")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", default="pathfinding_data")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-hud", action="store_true", help="Disable HUD overlay")
    parser.add_argument("--skip-test", action="store_true", help="Skip keyboard test")

    args = parser.parse_args()

    collector = DataCollector(
        output_dir=args.output,
        fps=args.fps,
        hud=not args.no_hud,
    )
    collector.collect(
        num_episodes=args.episodes,
        mode=args.mode,
        skip_test=args.skip_test,
    )
