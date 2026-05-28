#!/usr/bin/env python3
"""
v5.5 推理脚本 - 双头模型（Action + Mouse Bucket）【完全修复版】
修复：
1. 模型加载参数（num_goals=2）
2. Checkpoint 格式（提取 model_state_dict）
3. Frame stacking 维度（np.concatenate → [12, 224, 224]）
4. 鼠标执行逻辑（高频微步平滑）
"""
import argparse
import time
import numpy as np
import torch
import cv2
import pydirectinput as pdi
import pyautogui
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.goal_conditioned_bc_v55_optimized import GoalConditionedBC_v55


class WukongInferenceV55:
    """黑神话悟空 AI 推理器 v5.5（双头模型）【完全修复版】"""
    
    # Bucket 到 mouse_dx 的映射
    BUCKET_TO_DX = {
        0: -300,  # 快速左转
        1: -150,  # 中速左转
        2: -50,   # 慢速左转
        3: 0,     # 直行
        4: 50,    # 慢速右转
        5: 150,   # 中速右转
        6: 300,   # 快速右转
    }
    
    def __init__(self, model_path, goal_id=0, device="cuda:0", conf_threshold=0.5):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.goal_id = goal_id
        
        # 加载模型
        print(f"[加载] 模型: {model_path}")
        # 模型定义: GoalConditionedBC_v55(num_goals=2, freeze_backbone=False)
        self.model = GoalConditionedBC_v55(num_goals=2, freeze_backbone=False)
        
        # 修复：checkpoint 格式包含训练状态，需要提取 model_state_dict
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[加载] 从 checkpoint 加载模型状态 (epoch {checkpoint.get('epoch', '?')})")
        else:
            # 兼容旧格式（纯模型权重）
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"[加载] 模型已加载到 {self.device}")
        
        # 预处理变换
        self.preprocess = lambda frame: self._preprocess_frame(frame)
        
        # 状态跟踪
        self.w_pressed = False
        self.last_action = None
        self.frame_buffer = []
        self.last_mouse_time = 0
        self.last_dx = 0  # 平滑滤波
        
        print(f"[配置] Goal ID: {goal_id}")
        print(f"[配置] 置信度阈值: {conf_threshold}")
        print(f"[配置] 设备: {self.device}")
    
    def _preprocess_frame(self, frame):
        """预处理单帧"""
        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize → 224×224
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        # 归一化 + HWC → CHW
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_chw = np.transpose(frame_normalized, (2, 0, 1))  # [3, 224, 224]
        return frame_chw
    
    def preprocess_frame(self, frame):
        """公开预处理接口"""
        return self._preprocess_frame(frame)
    
    def predict(self, frames):
        """
        预测动作和鼠标转向
        frames: List[np.ndarray] 长度 4，每个元素 [H, W, 3] BGR
        """
        # 预处理：将 4 帧堆叠成 [12, 224, 224]（沿通道维度）
        processed = [self._preprocess_frame(f) for f in frames]  # List of [3, 224, 224]
        
        # 关键修复：沿通道维度堆叠（不是 batch 维度）
        frames_np = np.concatenate(processed, axis=0)  # [12, 224, 224] ✅
        frames_tensor = torch.from_numpy(frames_np).unsqueeze(0).to(self.device)  # [1, 12, 224, 224] ✅
        
        # Goal ID
        goal_ids = torch.tensor([self.goal_id], dtype=torch.long).to(self.device)
        
        # 推理
        with torch.no_grad():
            action_logits, mouse_logits = self.model(frames_tensor, goal_ids)
            
            # Action 预测
            action_probs = torch.softmax(action_logits, dim=-1)
            action_conf, action_idx = torch.max(action_probs, dim=-1)
            action_conf = action_conf.item()
            action_idx = action_idx.item()
            
            # Mouse Bucket 预测
            mouse_probs = torch.softmax(mouse_logits, dim=-1)
            mouse_conf, mouse_bucket = torch.max(mouse_probs, dim=-1)
            mouse_conf = mouse_conf.item()
            mouse_bucket = mouse_bucket.item()
        
        return action_idx, action_conf, mouse_bucket, mouse_conf
    
    def execute_action(self, action_idx, mouse_bucket):
        """执行预测的动作和鼠标转向"""
        # 动作映射
        action_map = {
            0: 'forward',
            1: 'turn_left',
            2: 'turn_right',
        }
        action_name = action_map.get(action_idx, 'forward')
        
        # 执行键盘动作
        if action_name == 'forward':
            if not self.w_pressed:
                pdi.keyDown('w')
                self.w_pressed = True
                print(f"[动作] 按住 W (forward)")
        else:
            if self.w_pressed:
                pdi.keyUp('w')
                self.w_pressed = False
            if action_name == 'turn_left':
                pdi.keyDown('a')
                time.sleep(0.05)
                pdi.keyUp('a')
            elif action_name == 'turn_right':
                pdi.keyDown('d')
                time.sleep(0.05)
                pdi.keyUp('d')
        
        # 执行鼠标转向（高频微步 + 平滑）
        current_time = time.time()
        if current_time - self.last_mouse_time >= 0.05:  # 50ms 间隔
            dx = self.BUCKET_TO_DX.get(mouse_bucket, 0)
            
            # 平滑滤波（指数移动平均）
            smoothed_dx = 0.7 * dx + 0.3 * self.last_dx
            self.last_dx = smoothed_dx
            
            # 分解大动作为微步（避免跳跃）
            steps = max(1, int(abs(smoothed_dx) / 10))
            micro_dx = smoothed_dx / steps
            
            for _ in range(steps):
                pdi.moveRel(int(micro_dx), 0, relative=True)
                time.sleep(0.01)  # 10ms 微步间隔
            
            self.last_mouse_time = current_time
        
        return action_name, mouse_bucket
    
    def run(self, duration=60):
        """运行推理"""
        print(f"\n[推理] 开始推理，时长: {duration} 秒")
        print(f"[推理] Goal ID: {self.goal_id}")
        print(f"[推理] 按 Ctrl+C 停止\n")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                # 截图
                frame = pyautogui.screenshot()
                frame = np.array(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 更新帧缓冲
                self.frame_buffer.append(frame)
                if len(self.frame_buffer) > 4:
                    self.frame_buffer.pop(0)
                
                # 只有 4 帧时才预测
                if len(self.frame_buffer) == 4:
                    action_idx, action_conf, mouse_bucket, mouse_conf = self.predict(self.frame_buffer)
                    
                    # 置信度过滤
                    if action_conf >= self.conf_threshold:
                        action_name, bucket_name = self.execute_action(action_idx, mouse_bucket)
                        frame_count += 1
                        
                        if frame_count % 10 == 0:
                            print(f"[推理] 帧 {frame_count}: {action_name} (conf={action_conf:.2f}), "
                                  f"bucket={mouse_bucket} (conf={mouse_conf:.2f})")
                
                # 控制频率
                time.sleep(0.05)  # 20 FPS
        
        except KeyboardInterrupt:
            print("\n[推理] 用户中断")
        finally:
            # 清理
            if self.w_pressed:
                pdi.keyUp('w')
            print(f"\n[推理] 推理结束，共 {frame_count} 帧")
        
        return frame_count


def main():
    parser = argparse.ArgumentParser(description='黑神话悟空 AI 推理 v5.5（双头模型）')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--goal-id', type=int, default=0, help='Goal ID (0 或 1)')
    parser.add_argument('--duration', type=int, default=60, help='推理时长（秒）')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    args = parser.parse_args()
    
    # 创建推理器
    inference = WukongInferenceV55(
        model_path=args.model,
        goal_id=args.goal_id,
        device=args.device,
        conf_threshold=args.conf_threshold
    )
    
    # 运行推理
    inference.run(duration=args.duration)


if __name__ == '__main__':
    main()
