"""
data_collector.py - 数据采集脚本 v2.1
录制人类玩家的游戏画面+操作，用于行为克隆和训练分析

改动说明（相比v2）：
1. 键盘监听：pynput → keyboard库（全局hook，游戏内可靠捕获ESC）
2. 停止方式：F1 → ESC（keyboard全局hook）+ --duration定时自动停
3. 自动保存：每--auto-save帧保存一次chunk，防止kill丢数据
4. HUD：默认关闭（--hud开启），去掉cv2依赖的无hud模式
5. 输入测试：简化为初始化检查，不需要cv2弹窗
6. 质量报告：录制结束后输出动作分布、鼠标活跃度、合格判定

使用方法：
  cd D:\projects\wukong_ai
  
  # 基本录制（5分钟，15fps，ESC停止或到时间自动停）
  C:\Python\python.exe -u training/data_collector.py --duration 300 --fps 15
  
  # 跳过输入测试（已确认环境OK时）
  C:\Python\python.exe -u training/data_collector.py --duration 300 --fps 15 --skip-test
  
  # 带HUD预览（需要前台终端，会弹出cv2窗口）
  C:\Python\python.exe -u training/data_collector.py --duration 300 --fps 15 --hud
  
  # 录完后检查结果（查看质量报告）
  C:\Python\python.exe training/data_collector.py --report pathfinding_data
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GAME_REGION, NUM_ACTIONS, ACTION_SPACE
from env.screen_capture import ScreenCapture

# 动作ID → 短名
ACTION_SHORT = {
    0: "idle", 1: "attack", 2: "heavy", 3: "dodge", 4: "forward",
    5: "right", 6: "left", 7: "dodge_atk", 8: "lock", 9: "heal",
}


class DataCollector:
    """人类游戏数据采集器 v2.1"""

    def __init__(self, output_dir="pathfinding_data", fps=15, hud=False):
        self.output_dir = output_dir
        self.fps = fps
        self.hud = hud
        self.capture = ScreenCapture()

        os.makedirs(output_dir, exist_ok=True)

        # 按键状态
        self._keys_pressed = set()
        self._recording = False

        # 鼠标状态（pynput）
        self._mouse_dx = 0
        self._mouse_dy = 0
        self._mouse_buttons = set()
        self._last_mouse_pos = None
        self._mouse_listener = None

    # ---- 键盘监听（keyboard库，全局hook） ----

    def setup_keyboard_listener(self):
        import keyboard as kb
        try:
            kb.unhook_all()
        except Exception:
            pass

        # 监听的按键
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
                # TODO: 改为F12可避免ESC同时打开游戏菜单
                # if name == 'f12' and e.event_type == kb.KEY_DOWN:
                #     self._recording = False

        kb.hook(on_key)
        return True

    # ---- 鼠标监听（pynput） ----

    def setup_mouse_listener(self):
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

        def on_click(x, y, button, pressed):
            name = button.name
            if pressed:
                self._mouse_buttons.add(name)
            else:
                self._mouse_buttons.discard(name)

        def on_scroll(x, y, dx, dy):
            self._mouse_dy += dy * 120

        self._mouse_listener = mouse.Listener(
            on_move=on_move, on_click=on_click, on_scroll=on_scroll
        )
        self._mouse_listener.start()
        return True

    def get_mouse_state(self):
        """获取本帧鼠标增量并重置"""
        dx = self._mouse_dx
        dy = self._mouse_dy
        buttons = frozenset(self._mouse_buttons)
        self._mouse_dx = 0
        self._mouse_dy = 0
        return dx, dy, buttons

    # ---- 按键→动作 ----

    def keys_to_action_id(self):
        keys = self._keys_pressed
        if 'j' in keys:
            return 7 if 'space' in keys else 1
        if 'space' in keys:
            return 3
        if 'w' in keys:
            return 4
        if 'd' in keys:
            return 5
        if 'a' in keys:
            return 6
        if 'v' in keys:
            return 8
        if 'r' in keys:
            return 9
        return 0

    # ---- 保存h5 ----

    def _save_h5(self, path, mode, frames, actions, timestamps,
                 mouse_dx, mouse_dy, mouse_buttons):
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset("frames", data=np.array(frames, dtype=np.uint8), compression="gzip")
            f.create_dataset("actions", data=np.array(actions, dtype=np.int8))
            f.create_dataset("timestamps", data=np.array(timestamps, dtype=np.float32))
            f.create_dataset("mouse_dx", data=np.array(mouse_dx, dtype=np.float32))
            f.create_dataset("mouse_dy", data=np.array(mouse_dy, dtype=np.float32))
            f.create_dataset("mouse_buttons", data=np.array(mouse_buttons, dtype=np.uint8))
            f.attrs["mode"] = mode
            f.attrs["fps"] = self.fps
            f.attrs["num_frames"] = len(frames)
            f.attrs["has_mouse"] = True
        sz = os.path.getsize(path) / 1024 / 1024
        print(f"  [SAVE] {os.path.basename(path)} ({len(frames)} frames, {sz:.1f}MB)", flush=True)

    # ---- 质量报告 ----

    def _print_report(self, all_results, output_dir):
        print(f"\n{'=' * 60}")
        print(f"  录制质量报告")
        print(f"{'=' * 60}")

        total_frames = 0
        total_time = 0.0
        total_action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        total_mouse_active = 0

        for i, r in enumerate(all_results):
            total_frames += r["frames"]
            total_time += r["duration"]
            total_action_counts += r["action_counts_full"]
            total_mouse_active += r["mouse_stats"]["frames_with_mouse"]

            print(f"\n  Episode {i+1}: {r['frames']} frames, {r['duration']:.1f}s, FPS={r['fps_actual']:.1f}")
            print(f"    动作分布:")
            for name, count in r["actions"].items():
                print(f"      {name}: {count} ({count/r['frames']*100:.1f}%)")
            ms = r["mouse_stats"]
            mouse_pct = ms["frames_with_mouse"] / r["frames"] * 100 if r["frames"] > 0 else 0
            print(f"    鼠标活跃: {ms['frames_with_mouse']}/{r['frames']} ({mouse_pct:.1f}%)")

        if total_frames == 0:
            print("\n  无有效数据！")
            return

        print(f"\n  {'─' * 40}")
        print(f"  总计: {total_frames} 帧, {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"\n  动作分布（汇总）:")
        for i in range(NUM_ACTIONS):
            count = int(total_action_counts[i])
            if count > 0:
                name = ACTION_SHORT[i]
                pct = count / total_frames * 100
                bar_len = 40
                filled = int(pct / 100 * bar_len)
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f"    {name:12s} {count:5d} ({pct:5.1f}%) {bar}")

        mouse_active_pct = total_mouse_active / total_frames * 100
        print(f"\n  鼠标活跃帧: {total_mouse_active}/{total_frames} ({mouse_active_pct:.1f}%)")

        # 合格判定
        print(f"\n  合格判定:")
        issues = []

        if total_frames < 3000:
            issues.append(f"数据量不足: {total_frames}帧 < 3000帧 (建议至少5分钟@10fps)")
        else:
            print(f"    ✅ 数据量: {total_frames}帧")

        idle_pct = total_action_counts[0] / total_frames * 100
        if idle_pct > 70:
            issues.append(f"idle占比过高: {idle_pct:.1f}% (建议<70%)")
        else:
            print(f"    ✅ idle占比: {idle_pct:.1f}%")

        active_actions = sum(1 for i in range(1, NUM_ACTIONS)
                           if total_action_counts[i] > total_frames * 0.005)
        if active_actions < 3:
            issues.append(f"动作类型少: {active_actions}种 (建议≥3种非idle动作)")
        else:
            print(f"    ✅ 动作多样性: {active_actions}种非idle动作")

        if mouse_active_pct < 5:
            issues.append(f"鼠标活跃度低: {mouse_active_pct:.1f}% (建议>5%)")
        else:
            print(f"    ✅ 鼠标活跃度: {mouse_active_pct:.1f}%")

        if issues:
            print(f"\n  ⚠️  问题:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"\n  🎉 录制质量合格！可以进行预处理和训练。")

        print(f"\n  下一步:")
        print(f"    1. 预处理: C:\\Python\\python.exe pathfinding/preprocess_data.py")
        print(f"    2. 训练:   C:\\Python\\python.exe pathfinding/behavior_clone_v2.py")
        print(f"    3. 推理:   C:\\Python\\python.exe pathfinding/inference_v2.py --duration 60 --fps 10")
        print(f"{'=' * 60}")

    # ---- 采集主循环 ----

    def collect(self, num_episodes=1, mode="pathfinding", skip_test=False,
                duration=300, auto_save=3000):
        import cv2

        print(f"\n{'=' * 60}", flush=True)
        print(f"  WukongAI Data Collector v2.1", flush=True)
        print(f"  Mode: {mode}  Episodes: {num_episodes}  FPS: {self.fps}", flush=True)
        print(f"  Duration: {duration}s  Auto-save: every {auto_save} frames", flush=True)
        print(f"  Output: {os.path.abspath(self.output_dir)}", flush=True)
        print(f"  Stop: ESC or duration timeout", flush=True)
        print(f"{'=' * 60}", flush=True)

        # Step 1: 输入设备检测（简化版，不需要cv2）
        if not skip_test:
            print(f"\n  检查输入设备...", flush=True)
            try:
                kb_ok = self.setup_keyboard_listener()
                print(f"  键盘(keyboard): {'OK' if kb_ok else 'FAILED'}", flush=True)
            except Exception as e:
                print(f"  键盘(keyboard): FAILED ({e})", flush=True)
                print(f"  请以管理员权限运行!", flush=True)
                return

            try:
                mouse_ok = self.setup_mouse_listener()
                time.sleep(0.5)
                mouse_alive = self._mouse_listener and self._mouse_listener.is_alive()
                print(f"  鼠标(pynput): {'OK' if mouse_alive else 'FAILED'}", flush=True)
            except Exception as e:
                print(f"  鼠标(pynput): FAILED ({e})", flush=True)
                mouse_alive = False

            # 清理，正式录制时重建
            import keyboard as kb
            kb.unhook_all()
            if self._mouse_listener:
                self._mouse_listener.stop()

            if not kb_ok or not mouse_alive:
                print(f"\n  输入设备异常！建议以管理员权限运行。", flush=True)
                return

        # Step 2: 正式录制
        print(f"\n{'=' * 60}", flush=True)
        print(f"  开始正式录制", flush=True)
        print(f"  切换到游戏窗口... (3秒倒计时)", flush=True)
        print(f"  按 ESC 停止  |  {duration}s后自动停止", flush=True)
        print(f"{'=' * 60}", flush=True)

        for i in range(3, 0, -1):
            print(f"  {i}...", flush=True)
            time.sleep(1)

        # 启动监听
        self.setup_keyboard_listener()
        self.setup_mouse_listener()
        self._keys_pressed.clear()
        self._last_mouse_pos = None

        all_results = []

        for ep in range(num_episodes):
            ep_frames = []
            ep_actions = []
            ep_timestamps = []
            ep_mouse_dx = []
            ep_mouse_dy = []
            ep_mouse_buttons = []
            saved_chunk_idx = 0

            self._recording = True
            print(f"\n  [Episode {ep+1}/{num_episodes}] RECORDING... (ESC to stop)", flush=True)

            start_time = time.time()
            frame_interval = 1.0 / self.fps

            while self._recording:
                loop_start = time.time()
                ep_elapsed = time.time() - start_time

                # 定时停止
                if duration > 0 and ep_elapsed >= duration:
                    print(f"  [TIME] {duration}s reached, stopping...", flush=True)
                    self._recording = False
                    break

                # 截图
                frame = self.capture.grab()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # 输入状态
                action_id = self.keys_to_action_id()
                mdx, mdy, mbtns = self.get_mouse_state()

                # HUD（可选）
                fps_actual = len(ep_frames) / max(ep_elapsed, 0.001)
                if self.hud:
                    hud_frame = self._draw_hud_simple(
                        frame, ep + 1, num_episodes,
                        len(ep_frames), fps_actual, action_id, ep_elapsed
                    )
                    cv2.imshow("WukongAI Recording", cv2.resize(hud_frame, (640, 360)))
                    cv2.waitKey(1)

                # 保存帧（缩小到224x224）
                small = cv2.resize(frame, (224, 224))
                ep_frames.append(small)
                ep_actions.append(action_id)
                ep_timestamps.append(time.time() - start_time)
                ep_mouse_dx.append(mdx)
                ep_mouse_dy.append(mdy)
                btn_val = 0
                if 'left' in mbtns: btn_val |= 1
                if 'right' in mbtns: btn_val |= 2
                if 'middle' in mbtns: btn_val |= 4
                ep_mouse_buttons.append(btn_val)

                # 自动保存chunk
                if auto_save > 0 and len(ep_frames) % auto_save == 0 and len(ep_frames) > 0:
                    saved_chunk_idx += 1
                    s = (saved_chunk_idx - 1) * auto_save
                    e_idx = saved_chunk_idx * auto_save
                    chunk_path = os.path.join(
                        self.output_dir,
                        f"{mode}_ep{ep+1}_chunk{saved_chunk_idx}_{int(time.time())}.h5"
                    )
                    self._save_h5(chunk_path, mode,
                        ep_frames[s:e_idx], ep_actions[s:e_idx],
                        ep_timestamps[s:e_idx], ep_mouse_dx[s:e_idx],
                        ep_mouse_dy[s:e_idx], ep_mouse_buttons[s:e_idx])

                # 进度（每5秒）
                if len(ep_frames) % (self.fps * 5) < 2 and len(ep_frames) > 0:
                    aname = ACTION_SHORT.get(action_id, "?")
                    print(f"  [{ep_elapsed:.0f}s] frames={len(ep_frames)} fps={fps_actual:.1f} action={aname}", flush=True)

                # 控制帧率
                dt = time.time() - loop_start
                if dt < frame_interval:
                    time.sleep(frame_interval - dt)

            # 停止后关HUD
            if self.hud:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            # 保存剩余帧
            remaining_start = saved_chunk_idx * auto_save if auto_save > 0 else 0
            if len(ep_frames) > max(remaining_start, 10):
                save_path = os.path.join(
                    self.output_dir,
                    f"{mode}_ep{ep+1}_{int(time.time())}.h5"
                )
                self._save_h5(save_path, mode,
                    ep_frames[remaining_start:],
                    ep_actions[remaining_start:],
                    ep_timestamps[remaining_start:],
                    ep_mouse_dx[remaining_start:],
                    ep_mouse_dy[remaining_start:],
                    ep_mouse_buttons[remaining_start:])

            # 统计
            if len(ep_frames) > 10:
                action_counts = np.bincount(np.array(ep_actions), minlength=NUM_ACTIONS)
                mouse_dx_arr = np.array(ep_mouse_dx)
                mouse_dy_arr = np.array(ep_mouse_dy)
                result = {
                    "frames": len(ep_frames),
                    "duration": ep_timestamps[-1],
                    "fps_actual": len(ep_frames) / ep_timestamps[-1],
                    "actions": {ACTION_SPACE[i][0]: int(action_counts[i])
                               for i in range(NUM_ACTIONS) if action_counts[i] > 0},
                    "action_counts_full": action_counts,
                    "mouse_stats": {
                        "total_dx": float(mouse_dx_arr.sum()),
                        "total_dy": float(mouse_dy_arr.sum()),
                        "frames_with_mouse": int(np.sum(
                            (np.abs(mouse_dx_arr) > 1) | (np.abs(mouse_dy_arr) > 1))),
                    },
                }
                all_results.append(result)
                print(f"\n  Episode {ep+1} done: {result['frames']} frames, "
                      f"{result['duration']:.1f}s, FPS={result['fps_actual']:.1f}", flush=True)
            else:
                print(f"  Too few frames ({len(ep_frames)}), skipped", flush=True)

            if ep < num_episodes - 1:
                print(f"\n  Next episode in 3s...", flush=True)
                time.sleep(3)

        # 清理
        import keyboard as kb
        try:
            kb.unhook_all()
        except Exception:
            pass
        if self._mouse_listener:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass
        self.capture.release()

        # 质量报告
        self._print_report(all_results, self.output_dir)
        return all_results

    # ---- 简化HUD ----

    def _draw_hud_simple(self, frame, ep, total_ep, frame_count,
                         fps_actual, action_id, duration):
        import cv2
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        blink = int(time.time() * 2) % 2 == 0
        rec_color = (0, 0, 255) if blink else (0, 0, 180)
        cv2.circle(frame, (25, 30), 6, rec_color, -1)
        cv2.putText(frame, f"REC Ep{ep}/{total_ep}  {frame_count}f  {fps_actual:.0f}fps  {duration:.0f}s  [ESC]",
                    (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 1)

        action_name = ACTION_SHORT.get(action_id, "?")
        cv2.putText(frame, f"Action: {action_name}",
                    (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        keys_str = ",".join(sorted(self._keys_pressed)) if self._keys_pressed else "-"
        cv2.putText(frame, f"Keys: {keys_str}",
                    (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WukongAI Data Collector v2.1")
    parser.add_argument("--mode", choices=["pathfinding", "combat"], default="pathfinding")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", default="pathfinding_data")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--duration", type=int, default=300, help="每episode最长秒数(0=不限)")
    parser.add_argument("--auto-save", type=int, default=3000, help="每N帧自动保存(0=结束才保存)")
    parser.add_argument("--hud", action="store_true", help="启用HUD预览窗口")
    parser.add_argument("--skip-test", action="store_true", help="跳过输入设备测试")

    args = parser.parse_args()

    collector = DataCollector(
        output_dir=args.output,
        fps=args.fps,
        hud=args.hud,
    )
    collector.collect(
        num_episodes=args.episodes,
        mode=args.mode,
        skip_test=args.skip_test,
        duration=args.duration,
        auto_save=args.auto_save,
    )
