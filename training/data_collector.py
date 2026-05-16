"""
data_collector.py - 数据采集脚本
录制人类玩家的游戏画面+操作，用于行为克隆和训练分析
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GAME_REGION, NUM_ACTIONS, ACTION_SPACE
from env.screen_capture import ScreenCapture


class DataCollector:
    """人类游戏数据采集器"""
    
    def __init__(self, output_dir="pathfinding_data", fps=30):
        self.output_dir = output_dir
        self.fps = fps
        self.capture = ScreenCapture()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 按键监听
        self._recording = False
        self._keys_pressed = set()
    
    def setup_keyboard_listener(self):
        """设置键盘监听"""
        from pynput import keyboard
        
        def on_press(key):
            try:
                self._keys_pressed.add(key.char.lower())
            except AttributeError:
                if key == keyboard.Key.space:
                    self._keys_pressed.add('space')
                elif key == keyboard.Key.shift:
                    self._keys_pressed.add('shift')
        
        def on_release(key):
            try:
                self._keys_pressed.discard(key.char.lower())
            except AttributeError:
                if key == keyboard.Key.space:
                    self._keys_pressed.discard('space')
                elif key == keyboard.Key.shift:
                    self._keys_pressed.discard('shift')
            
            # F1停止录制
            if key == keyboard.Key.f1:
                self._recording = False
                return False
        
        self._listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
        )
        self._listener.start()
    
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
    
    def collect(self, num_episodes=5, mode="pathfinding"):
        """
        录制数据
        
        Args:
            num_episodes: 录制几个episode
            mode: "pathfinding" | "combat" - 不同模式可能使用不同的动作映射
        """
        import h5py
        import cv2
        
        print(f"\n{'='*60}")
        print(f"  数据采集模式: {mode}")
        print(f"  目标: {num_episodes} 个episode")
        print(f"  按 F1 停止录制")
        print(f"  输出目录: {self.output_dir}")
        print(f"{'='*60}\n")
        
        self.setup_keyboard_listener()
        time.sleep(2)  # 等待切换到游戏窗口
        
        for ep in range(num_episodes):
            episode_frames = []
            episode_actions = []
            episode_timestamps = []
            
            print(f"[Episode {ep+1}/{num_episodes}] 开始录制... 按F1停止")
            self._recording = True
            
            start_time = time.time()
            frame_interval = 1.0 / self.fps
            
            while self._recording:
                loop_start = time.time()
                
                # 截图
                frame = self.capture.grab()
                
                # 获取动作
                action_id = self.keys_to_action_id()
                
                # 保存
                # 缩小帧以节省空间（从1920x1080到224x224）
                small_frame = cv2.resize(frame, (224, 224))
                
                episode_frames.append(small_frame)
                episode_actions.append(action_id)
                episode_timestamps.append(time.time() - start_time)
                
                # 控制帧率
                elapsed = time.time() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
            
            # 保存episode数据
            if len(episode_frames) > 10:  # 至少10帧才保存
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
                print(f"  已保存: {save_path}")
                print(f"  帧数: {len(episode_frames)}")
                print(f"  时长: {episode_timestamps[-1]:.1f}s")
                for i, count in enumerate(action_counts):
                    if count > 0:
                        print(f"    {ACTION_SPACE[i][0]}: {count} ({count/len(episode_frames)*100:.1f}%)")
            else:
                print(f"  帧数太少({len(episode_frames)})，跳过保存")
            
            if ep < num_episodes - 1:
                print(f"\n准备下一个episode... (3秒后开始)")
                time.sleep(3)
                self._recording = True
        
        self._listener.stop()
        self.capture.release()
        print(f"\n[DataCollector] 采集完成！数据保存在: {self.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="黑神话悟空 - 数据采集")
    parser.add_argument("--mode", choices=["pathfinding", "combat"], default="pathfinding")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", default="pathfinding_data")
    parser.add_argument("--fps", type=int, default=30)
    
    args = parser.parse_args()
    
    collector = DataCollector(output_dir=args.output, fps=args.fps)
    collector.collect(num_episodes=args.episodes, mode=args.mode)
