"""
data_collector.py - 数据采集脚本（增强版 v2）
录制人类玩家的游戏画面+操作，用于行为克隆和训练分析

增强功能：
- 屏幕左上角实时HUD叠加显示录制状态
- 键盘 + 鼠标监听（dx/dy + 按键）
- 启动时检测输入设备是否正常捕获
- F1停止 + 可视化提示
- 实时帧率/动作名称/鼠标状态显示
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
    """人类游戏数据采集器（含鼠标）"""

    def __init__(self, output_dir="pathfinding_data", fps=30, hud=True):
        self.output_dir = output_dir
        self.fps = fps
        self.hud = hud
        self.capture = ScreenCapture()

        os.makedirs(output_dir, exist_ok=True)

        # 按键监听
        self._recording = False
        self._keys_pressed = set()
        self._key_log = []

        # 鼠标监听
        self._mouse_dx = 0
        self._mouse_dy = 0
        self._mouse_buttons = set()
        self._mouse_log = []
        self._last_mouse_pos = None

    # ---- 键盘监听 ----

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

        self._keyboard_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
        )
        self._keyboard_listener.start()

    # ---- 鼠标监听 ----

    def setup_mouse_listener(self):
        """设置鼠标监听（移动+点击+滚轮）"""
        from pynput import mouse

        def on_move(x, y):
            if self._last_mouse_pos is None:
                self._last_mouse_pos = (x, y)
                return
            dx = x - self._last_mouse_pos[0]
            dy = y - self._last_mouse_pos[1]
            self._last_mouse_pos = (x, y)
            if abs(dx) > 0 or abs(dy) > 0:
                self._mouse_dx += dx
                self._mouse_dy += dy
                self._mouse_log.append(f"MV:{dx:+.0f},{dy:+.0f}")
                if len(self._mouse_log) > 50:
                    self._mouse_log = self._mouse_log[-50:]

        def on_click(x, y, button, pressed):
            name = button.name  # 'left', 'right', 'middle'
            if pressed:
                self._mouse_buttons.add(name)
                self._mouse_log.append(f"DN:{name}")
            else:
                self._mouse_buttons.discard(name)
                self._mouse_log.append(f"UP:{name}")
            if len(self._mouse_log) > 50:
                self._mouse_log = self._mouse_log[-50:]

        def on_scroll(x, y, dx, dy):
            self._mouse_dy += dy * 120  # 标准化滚轮
            self._mouse_log.append(f"SC:{dy:+d}")
            if len(self._mouse_log) > 50:
                self._mouse_log = self._mouse_log[-50:]

        self._mouse_listener = mouse.Listener(
            on_move=on_move,
            on_click=on_click,
            on_scroll=on_scroll,
        )
        self._mouse_listener.start()

    def get_mouse_state(self):
        """获取本帧鼠标状态并重置累积量（每帧调用一次）"""
        dx = self._mouse_dx
        dy = self._mouse_dy
        buttons = frozenset(self._mouse_buttons)
        self._mouse_dx = 0
        self._mouse_dy = 0
        return dx, dy, buttons

    # ---- 输入测试 ----

    def test_inputs(self):
        """测试键盘+鼠标监听是否正常工作"""
        import cv2

        print("\n" + "=" * 50)
        print("  输入设备测试")
        print("  请依次测试:")
        print("    键盘: W  A  S  D  Space  J")
        print("    鼠标: 移动鼠标  左/右键点击")
        print("    F1 停止测试")
        print("=" * 50)

        self.setup_keyboard_listener()
        self.setup_mouse_listener()
        self._recording = True

        start = time.time()
        while self._recording and time.time() - start < 15:
            frame = self.capture.grab()
            if frame is None:
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            overlay = frame.copy()

            keys_str = ", ".join(sorted(self._keys_pressed)) if self._keys_pressed else "(none)"
            mouse_str = f"dx:{self._mouse_dx:+.0f} dy:{self._mouse_dy:+.0f} btns:{'+'.join(sorted(self._mouse_buttons)) if self._mouse_buttons else 'none'}"
            recent_keys = self._key_log[-4:]
            recent_mouse = self._mouse_log[-3:]

            # 半透明背景
            cv2.rectangle(overlay, (10, 10), (550, 280), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            cv2.putText(frame, "[INPUT TEST] Keyboard + Mouse / F1 to stop",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.putText(frame, f"Keys: {keys_str}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            y = 105
            for log in recent_keys:
                color = (0, 200, 255) if log.startswith("DN") else (100, 100, 100)
                cv2.putText(frame, log, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                y += 20

            y += 8
            cv2.putText(frame, f"Mouse: {mouse_str}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
            y += 25
            for log in recent_mouse:
                cv2.putText(frame, log, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
                y += 20

            cv2.imshow("WukongAI - Input Test", cv2.resize(frame, (640, 360)))
            cv2.waitKey(1)
            time.sleep(0.03)

        cv2.destroyAllWindows()
        self._keyboard_listener.stop()
        self._mouse_listener.stop()

        captured_keys = len([l for l in self._key_log if l.startswith("DN")])
        captured_mouse = len([l for l in self._mouse_log if l.startswith("MV")])

        if captured_keys > 0 and captured_mouse > 0:
            print(f"\n  OK! 键盘: {captured_keys}次按键, 鼠标: {captured_mouse}次移动")
            return True
        else:
            print(f"\n  WARNING: 键盘={captured_keys}次, 鼠标={captured_mouse}次")
            if captured_keys == 0:
                print(f"  键盘未捕获! 请检查输入法(需ENG)或管理员权限")
            if captured_mouse == 0:
                print(f"  鼠标未捕获! 请检查管理员权限或杀毒软件拦截")
            return False

    def reset_input_state(self):
        """重置所有输入状态并重建监听器"""
        self._keys_pressed = set()
        self._key_log = []
        self._mouse_dx = 0
        self._mouse_dy = 0
        self._mouse_buttons = set()
        self._mouse_log = []
        self._last_mouse_pos = None
        self.setup_keyboard_listener()
        self.setup_mouse_listener()

    # ---- 按键→动作映射 ----

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

    # ---- HUD ----

    def draw_hud(self, frame, ep, total_ep, frame_count, fps_actual,
                 action_id, duration, keys_pressed,
                 mouse_dx=0, mouse_dy=0, mouse_buttons=None):
        """在帧上绘制HUD叠加层"""
        import cv2

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 半透明黑色背景 (左上角) - 增加高度容纳鼠标行
        bar_w = 520
        bar_h = 220
        cv2.rectangle(overlay, (10, 10), (10 + bar_w, 10 + bar_h), (0, 0, 0), -1)
        result = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        y = 38
        # 录制状态
        blink = int(time.time() * 2) % 2 == 0
        rec_color = (0, 0, 255) if blink else (0, 0, 180)
        cv2.circle(result, (25, y - 5), 8, rec_color, -1)
        cv2.putText(result, "REC", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, rec_color, 2)

        cv2.putText(result, f"Episode: {ep}/{total_ep}",
                     (110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(result, "[F1] STOP",
                     (330, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        y += 30
        cv2.putText(result, f"Frames: {frame_count}  FPS: {fps_actual:.1f}  Time: {duration:.1f}s",
                     (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        y += 26
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

        y += 26
        # 当前按键
        keys_str = ", ".join(sorted(keys_pressed)) if keys_pressed else "(none)"
        cv2.putText(result, f"Keys: {keys_str}",
                     (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        y += 22
        # 鼠标状态
        btn_str = "+".join(sorted(mouse_buttons)) if mouse_buttons else "none"
        cv2.putText(result, f"Mouse: dx={mouse_dx:+.0f} dy={mouse_dy:+.0f} [{btn_str}]",
                     (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

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
                cv2.rectangle(result, (bar_x, bar_y), (bar_x + 150, bar_y + 14), (50, 50, 50), -1)
                bar_fill = int(150 * pct)
                colors = [(100,100,100),(0,100,255),(0,80,200),(0,255,255),(0,200,0),(0,200,0),(0,200,0),(255,100,0),(200,200,0),(0,255,0)]
                cv2.rectangle(result, (bar_x, bar_y), (bar_x + bar_fill, bar_y + 14), colors[i], -1)
                short_names = ["idle","atk","heavy","dodge","fwd","rgt","lft","d_atk","lock","heal"]
                cv2.putText(result, f"{short_names[i]}:{count}",
                             (bar_x + 3, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
                bar_y += 17

        return result

    # ---- 数据采集主循环 ----

    def collect(self, num_episodes=5, mode="pathfinding", skip_test=False):
        """
        录制数据

        Args:
            num_episodes: 录制几个episode
            mode: "pathfinding" | "combat"
            skip_test: 跳过输入设备测试
        """
        import h5py
        import cv2

        print(f"\n{'=' * 60}")
        print(f"  WukongAI Data Collector (Enhanced v2)")
        print(f"  Mode: {mode}  |  Episodes: {num_episodes}  |  FPS: {self.fps}")
        print(f"  Output: {os.path.abspath(self.output_dir)}")
        print(f"  [NEW] Mouse recording enabled (dx/dy + buttons)")
        print(f"{'=' * 60}")

        # Step 1: 输入设备测试
        if not skip_test:
            print()
            ok = self.test_inputs()
            if not ok:
                print("\n  输入设备异常！建议以管理员权限运行。")
                choice = input("  是否继续? (y/n): ").strip().lower()
                if choice != 'y':
                    print("  已退出。")
                    return
            # 重建监听器（测试时被停止了）
            self.reset_input_state()

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
            episode_mouse_dx = []
            episode_mouse_dy = []
            episode_mouse_buttons = []

            self._action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
            self._recording = True

            print(f"\n  [Episode {ep+1}/{num_episodes}] RECORDING... (F1 to stop)")

            start_time = time.time()
            frame_interval = 1.0 / self.fps

            while self._recording:
                loop_start = time.time()

                # 截图
                frame = self.capture.grab()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # 获取动作 + 鼠标状态
                action_id = self.keys_to_action_id()
                mdx, mdy, mbtns = self.get_mouse_state()

                # 绘制HUD
                duration = time.time() - start_time
                fps_actual = len(episode_frames) / max(duration, 0.001)

                if self.hud:
                    frame = self.draw_hud(
                        frame, ep + 1, num_episodes,
                        len(episode_frames), fps_actual,
                        action_id, duration, self._keys_pressed,
                        mouse_dx=mdx, mouse_dy=mdy, mouse_buttons=mbtns
                    )
                    preview = cv2.resize(frame, (640, 360))
                    cv2.imshow("WukongAI - Recording", preview)
                    cv2.waitKey(1)

                # 缩小帧保存
                small_frame = cv2.resize(frame, (224, 224))
                episode_frames.append(small_frame)
                episode_actions.append(action_id)
                episode_timestamps.append(time.time() - start_time)

                # 鼠标数据
                episode_mouse_dx.append(mdx)
                episode_mouse_dy.append(mdy)
                # 按键编码: left=1, right=2, middle=4
                btn_val = 0
                if 'left' in mbtns:
                    btn_val |= 1
                if 'right' in mbtns:
                    btn_val |= 2
                if 'middle' in mbtns:
                    btn_val |= 4
                episode_mouse_buttons.append(btn_val)

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
                    f.create_dataset(
                        "mouse_dx",
                        data=np.array(episode_mouse_dx, dtype=np.float32),
                    )
                    f.create_dataset(
                        "mouse_dy",
                        data=np.array(episode_mouse_dy, dtype=np.float32),
                    )
                    f.create_dataset(
                        "mouse_buttons",
                        data=np.array(episode_mouse_buttons, dtype=np.uint8),
                    )
                    f.attrs["mode"] = mode
                    f.attrs["fps"] = self.fps
                    f.attrs["num_frames"] = len(episode_frames)
                    f.attrs["has_mouse"] = True

                # 统计
                action_counts = np.bincount(
                    np.array(episode_actions),
                    minlength=NUM_ACTIONS,
                )
                mouse_dx_arr = np.array(episode_mouse_dx)
                mouse_dy_arr = np.array(episode_mouse_dy)
                result = {
                    "path": save_path,
                    "frames": len(episode_frames),
                    "duration": episode_timestamps[-1],
                    "fps_actual": len(episode_frames) / episode_timestamps[-1],
                    "actions": {ACTION_SPACE[i][0]: int(action_counts[i]) for i in range(NUM_ACTIONS) if action_counts[i] > 0},
                    "mouse_stats": {
                        "total_dx": float(mouse_dx_arr.sum()),
                        "total_dy": float(mouse_dy_arr.sum()),
                        "frames_with_mouse": int(np.sum((np.abs(mouse_dx_arr) > 1) | (np.abs(mouse_dy_arr) > 1))),
                    },
                }
                all_results.append(result)

                print(f"\n  SAVED: {os.path.basename(save_path)}")
                print(f"  Frames: {len(episode_frames)}  Duration: {episode_timestamps[-1]:.1f}s  FPS: {result['fps_actual']:.1f}")
                for name, count in result["actions"].items():
                    print(f"    {name}: {count} ({count/len(episode_frames)*100:.1f}%)")
                ms = result["mouse_stats"]
                print(f"    Mouse: dx={ms['total_dx']:+.0f} dy={ms['total_dy']:+.0f} active_frames={ms['frames_with_mouse']}")
            else:
                print(f"  Too few frames ({len(episode_frames)}), skipped")

            if ep < num_episodes - 1:
                print(f"\n  Next episode in 3s...")
                time.sleep(3)

        self._keyboard_listener.stop()
        self._mouse_listener.stop()
        self.capture.release()

        # 总结
        print(f"\n{'=' * 60}")
        print(f"  DONE! {len(all_results)} episodes saved to:")
        print(f"  {os.path.abspath(self.output_dir)}")
        total_frames = sum(r["frames"] for r in all_results)
        total_time = sum(r["duration"] for r in all_results)
        print(f"  Total: {total_frames} frames, {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"  Mouse data: mouse_dx, mouse_dy, mouse_buttons in each h5")
        print(f"{'=' * 60}")

        return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WukongAI - Enhanced Data Collector v2 (with mouse)")
    parser.add_argument("--mode", choices=["pathfinding", "combat"], default="pathfinding")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", default="pathfinding_data")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-hud", action="store_true", help="Disable HUD overlay")
    parser.add_argument("--skip-test", action="store_true", help="Skip input device test")

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
