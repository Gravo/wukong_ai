"""
pointnav_data_collector.py - PointNav专用数据采集器
=====================================================
采集格式：(当前帧, 目标帧, 动作, 鼠标移动)

目标帧来源：
  1. 手动标记存档点画面（录制时按数字键）
  2. 自动提取轨迹末端作为目标
  3. 手动截取存档点截图

使用方法：
  python data_collection_pointnav.py --mode record --goal-dir savepoints
"""

import sys
import os
import time
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import torch
import cv2
import win32gui
import win32ui
import win32con
from ctypes import windll
from PIL import Image

from config import (
    GAME_WINDOW_TITLE, GAME_REGION, FRAME_WIDTH, FRAME_HEIGHT,
    ACTION_SPACE, KEY_MAP, NUM_ACTIONS
)
from models.pointnav_model import WukongPointNav, DepthEstimator


class PointNavDataCollector:
    """
    PointNav专用数据采集器
    
    采集数据格式：
      - frames: (T, 3, H, W) 当前帧序列
      - goal_frame: (3, H, W) 目标帧（存档点画面）
      - actions: (T,) 动作ID序列
      - mouse_dx/dy: (T,) 鼠标移动序列
    
    数据命名规则：
      {timestamp}_{start_region}_{end_region}.h5
      例: 20260520_143000_savepoint_A_to_B.h5
    """
    
    def __init__(
        self,
        output_dir="pathfinding_data",
        goal_dir="savepoint_frames",
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT
    ):
        self.output_dir = output_dir
        self.goal_dir = goal_dir
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(goal_dir, exist_ok=True)
        
        # 状态
        self.frames = []
        self.actions = []
        self.mouse_dx = []
        self.mouse_dy = []
        self.goal_frame = None
        self.goal_saved = False
        
        # 键盘状态
        self.key_states = {}
        self.last_mouse_pos = None
        
        # 截图器
        self.screenshotter = Screenshotter(GAME_WINDOW_TITLE, GAME_REGION)
        
        # 深度估计
        self.depth_estimator = DepthEstimator()
        
        print(f"PointNav Data Collector initialized", flush=True)
        print(f"  Output: {output_dir}", flush=True)
        print(f"  Goals: {goal_dir}", flush=True)
    
    def save_goal_frame(self, frame=None, name=None):
        """
        保存目标帧（存档点画面）
        
        Args:
            frame: 要保存的帧，不提供则截取当前画面
            name: 保存名称，不提供则用时间戳
        """
        if frame is None:
            frame = self.screenshotter.capture()
        
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        path = os.path.join(self.goal_dir, f"{name}.png")
        
        # 转换保存
        if isinstance(frame, np.ndarray):
            img = Image.fromarray(frame.astype(np.uint8))
        else:
            img = frame
        
        img.save(path)
        self.goal_frame = self._preprocess_frame(frame)
        self.goal_saved = True
        
        print(f"Goal saved: {path}", flush=True)
        return path
    
    def load_goal_frames(self):
        """加载所有已保存的目标帧"""
        goal_files = glob.glob(os.path.join(self.goal_dir, "*.png"))
        goals = []
        for f in goal_files:
            img = Image.open(f).convert("RGB")
            img = img.resize((self.frame_width, self.frame_height))
            goals.append({
                "name": os.path.splitext(os.path.basename(f))[0],
                "frame": self._preprocess_frame(np.array(img)),
                "path": f
            })
        return goals
    
    def _preprocess_frame(self, frame):
        """预处理帧"""
        if frame is None:
            return None
        
        # resize
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # BGR -> RGB
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 归一化到[0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # (H, W, C) -> (C, H, W)
        frame = frame.transpose(2, 0, 1)
        
        return frame
    
    def start_recording(self, goal_frame=None):
        """开始录制一段轨迹"""
        self.frames = []
        self.actions = []
        self.mouse_dx = []
        self.mouse_dy = []
        
        if goal_frame is not None:
            self.goal_frame = self._preprocess_frame(goal_frame)
            self.goal_saved = True
        
        self.last_mouse_pos = win32gui.GetCursorPos()
        print("Recording started", flush=True)
    
    def record_step(self, action_id=0):
        """记录一步"""
        # 截图
        frame = self.screenshotter.capture()
        frame = self._preprocess_frame(frame)
        
        # 鼠标移动
        current_pos = win32gui.GetCursorPos()
        if self.last_mouse_pos:
            dx = current_pos[0] - self.last_mouse_pos[0]
            dy = current_pos[1] - self.last_mouse_pos[1]
        else:
            dx, dy = 0, 0
        self.last_mouse_pos = current_pos
        
        # 记录
        self.frames.append(frame)
        self.actions.append(action_id)
        self.mouse_dx.append(dx)
        self.mouse_dy.append(dy)
    
    def stop_recording(self, trajectory_name=None):
        """
        停止录制并保存
        
        Args:
            trajectory_name: 轨迹名称
        """
        if not self.frames:
            print("No frames recorded", flush=True)
            return None
        
        # 创建数据
        frames = np.array(self.frames, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int64)
        mouse_dx = np.array(self.mouse_dx, dtype=np.float32)
        mouse_dy = np.array(self.mouse_dy, dtype=np.float32)
        
        # 保存
        if trajectory_name is None:
            trajectory_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{trajectory_name}.h5"
        filepath = os.path.join(self.output_dir, filename)
        
        with h5py.File(filepath, "w") as f:
            f.create_dataset("frames", data=frames)
            f.create_dataset("actions", data=actions)
            f.create_dataset("mouse_dx", data=mouse_dx)
            f.create_dataset("mouse_dy", data=mouse_dy)
            
            # 保存目标帧（如果有）
            if self.goal_frame is not None:
                f.create_dataset("goal_frame", data=self.goal_frame)
            
            # 元数据
            f.attrs["trajectory_name"] = trajectory_name
            f.attrs["num_frames"] = len(frames)
            f.attrs["goal_saved"] = self.goal_saved
            f.attrs["timestamp"] = datetime.now().isoformat()
        
        print(f"Saved: {filepath} ({len(frames)} frames)", flush=True)
        
        # 重置
        self.frames = []
        self.actions = []
        self.mouse_dx = []
        self.mouse_dy = []
        
        return filepath
    
    def create_goal_from_trajectory(self, frames, goal_idx=None):
        """
        从轨迹中创建目标帧
        
        Args:
            frames: 帧序列
            goal_idx: 目标帧索引，None则用最后一帧
        """
        if goal_idx is None:
            goal_idx = -1
        
        return frames[goal_idx]
    
    def augment_with_goals(self, h5_path):
        """
        为已有数据添加目标帧
        
        用于后期处理：加载已有h5，自动生成goal_frame
        """
        with h5py.File(h5_path, "r+") as f:
            frames = f["frames"][:]
            T = len(frames)
            
            # 生成目标帧序列（每个帧对应的目标）
            goal_indices = []
            for i in range(T):
                if T - i > 5:
                    goal_indices.append(min(i + (T - i) // 2, T - 1))
                else:
                    goal_indices.append(T - 1)
            
            goal_frames = frames[goal_indices]
            
            if "goal_frames" not in f:
                f.create_dataset("goal_frames", data=goal_frames)
                print(f"Added goal_frames to {h5_path}", flush=True)
            else:
                print(f"goal_frames already exists in {h5_path}", flush=True)


# ============================================================
# 屏幕截图器（从data_collector.py移植）
# ============================================================

class Screenshotter:
    """游戏画面截图器"""
    
    def __init__(self, window_title=GAME_WINDOW_TITLE, region=None):
        self.window_title = window_title
        self.region = region
        
        # 找窗口
        self.hwnd = win32gui.FindWindow(None, window_title)
        if not self.hwnd:
            raise ValueError(f"Window not found: {window_title}")
        
        # 获取DC
        self.hwndDC = win32gui.GetWindowDC(self.hwnd)
        self.mfcDC = win32ui.CreateDCFromHandle(self.hwndDC)
        self.saveDC = self.mfcDC.CreateCompatibleDC()
        
        # 窗口区域
        if region:
            self.left = region["left"]
            self.top = region["top"]
            self.width = region["width"]
            self.height = region["height"]
        else:
            self.left, self.top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            self.width = right - self.left
            self.height = bottom - self.top
        
        # 创建bitmap
        self.saveBitMap = win32ui.CreateBitmap()
        self.saveBitMap.CreateCompatibleBitmap(self.mfcDC, self.width, self.height)
        self.saveDC.SelectObject(self.saveBitMap)
        
        # 屏幕DC
        self.GetDC = windll.user32.GetDC
        self.ReleaseDC = windll.user32.ReleaseDC
        
        print(f"Screenshotter initialized: {self.width}x{self.height}", flush=True)
    
    def capture(self):
        """截取当前画面"""
        # 复制窗口内容
        result = windll.gdi32.BitBlt(
            self.saveDC.GetSafeHdc(),
            0, 0, self.width, self.height,
            self.hwndDC,
            0, 0,
            win32con.SRCCOPY
        )
        
        # 转换
        bmpinfo = self.saveBitMap.GetInfo()
        bmpstr = self.saveBitMap.GetBitmapBits(True)
        
        img = np.frombuffer(bmpstr, dtype=np.uint8)
        img = img.reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    
    def __del__(self):
        """清理"""
        try:
            win32gui.DeleteObject(self.saveBitMap.GetHandle())
            self.saveDC.DeleteDC()
            self.mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, self.hwndDC)
        except:
            pass


# ============================================================
# 主入口
# ============================================================

def main():
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="PointNav Data Collector")
    parser.add_argument("--mode", choices=["record", "augment"], default="record",
                       help="record: 录制新数据 | augment: 为已有数据添加goal_frame")
    parser.add_argument("--output-dir", default="pathfinding_data")
    parser.add_argument("--goal-dir", default="savepoint_frames")
    parser.add_argument("--goal-image", default=None,
                       help="手动指定目标画面路径")
    args = parser.parse_args()
    
    collector = PointNavDataCollector(
        output_dir=args.output_dir,
        goal_dir=args.goal_dir
    )
    
    if args.mode == "record":
        print("=" * 50, flush=True)
        print("PointNav Data Recording Mode", flush=True)
        print("=" * 50, flush=True)
        print("Controls:", flush=True)
        print("  数字键1-9: 标记存档点为目标", flush=True)
        print("  R: 开始/停止录制", flush=True)
        print("  S: 保存当前轨迹", flush=True)
        print("  Q: 退出", flush=True)
        print("=" * 50, flush=True)
        
        # 加载已有关键帧
        existing_goals = collector.load_goal_frames()
        print(f"Loaded {len(existing_goals)} existing goal frames", flush=True)
        
        recording = False
        trajectory_count = 0
        
        # 简单循环（实际使用时建议用独立线程）
        print("Ready. Press R to start recording.", flush=True)
        
        # 保存当前目标
        if args.goal_image:
            from PIL import Image
            img = Image.open(args.goal_image)
            collector.save_goal_frame(np.array(img), "manual_goal")
    
    elif args.mode == "augment":
        print("=" * 50, flush=True)
        print("PointNav Data Augmentation Mode", flush=True)
        print("=" * 50, flush=True)
        
        h5_files = glob.glob(os.path.join(args.output_dir, "*.h5"))
        print(f"Found {len(h5_files)} h5 files", flush=True)
        
        for h5_path in h5_files:
            try:
                collector.augment_with_goals(h5_path)
            except Exception as e:
                print(f"Error: {h5_path}: {e}", flush=True)
        
        print("Augmentation complete!", flush=True)


if __name__ == "__main__":
    main()
