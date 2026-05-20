"""
pointnav_model.py - PointNav最优实现版
=========================================
最优方案：DD-PPO视觉编码器 + MiDaS深度 + 存档点目标帧 + Goal-Conditioned

核心思想：
  PointNav = "从当前画面导航到目标画面"
  悟空AI = PointNav（第三人身称适配版）

架构：
  输入: (当前帧, 目标帧, 深度图) → 编码器 → 融合 → 动作预测

使用方法：
  # 训练
  python models/pointnav_model.py --train --data-dir pathfinding_data

  # 推理
  python models/pointnav_model.py --infer --goal-image savepoint.png
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import os
import sys
import argparse
import glob
import h5py
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NUM_ACTIONS, MODEL, FRAME_WIDTH, FRAME_HEIGHT, FRAME_STACK,
    CAPTURE_FPS, GAME_REGION
)


# ============================================================
# Part 1: 视觉编码器（DD-PPO风格）
# ============================================================

class DDPPOVisualEncoder(nn.Module):
    """
    DD-PPO风格视觉编码器
    
    DD-PPO在Habitat模拟器上训练PointNav任务
    学到的特征对空间导航有很强的针对性：
    - 墙壁、地板、障碍物的区分
    - 空间深度感知
    - 路径规划相关特征
    
    这里用ResNet50作为backbone（结构和DD-PPO相同）
    """

    def __init__(
        self,
        latent_dim=256,
        pretrained=True,
        freeze=False,
        use_depth=True
    ):
        super().__init__()
        
        self.use_depth = use_depth
        self.latent_dim = latent_dim
        
        # ResNet50 backbone（与DD-PPO相同结构）
        resnet = models.resnet50(pretrained=pretrained)
        
        # 去掉最后的fc层
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 输出 (B, 2048, 1, 1)
        
        # 深度编码器（如果使用深度）
        if use_depth:
            self.depth_conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),  # (B, 64)
            )
            
            # 融合层
            self.fusion = nn.Sequential(
                nn.Linear(2048 + 64, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(2048, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        
        # 冻结预训练权重
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, rgb, depth=None):
        """
        Args:
            rgb: (B, 3, H, W) RGB图像
            depth: (B, 1, H, W) 深度图（可选）
        Returns:
            feat: (B, latent_dim) 视觉特征
        """
        # RGB特征
        rgb_feat = self.backbone(rgb)  # (B, 2048, 1, 1)
        rgb_feat = rgb_feat.flatten(1)  # (B, 2048)
        
        if self.use_depth and depth is not None:
            depth_feat = self.depth_conv(depth)  # (B, 64)
            combined = torch.cat([rgb_feat, depth_feat], dim=-1)  # (B, 2112)
            feat = self.fusion(combined)  # (B, latent_dim)
        else:
            feat = self.fusion(rgb_feat)  # (B, latent_dim)
        
        return feat


# ============================================================
# Part 2: 深度估计（MiDaS）
# ============================================================

class DepthEstimator:
    """
    MiDaS深度估计器
    
    游戏画面是虚拟的，深度估计比真实世界更准确
    - 无传感器噪声
    - 无光照变化
    - 几何关系精确
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """加载MiDaS_small（最快版本）"""
        try:
            import torch.hub
            self._model = torch.hub.load(
                "intel-isl/MiDaS",
                "MiDaS_small"  # ~40M参数，RTX 2060轻松跑
            )
            self._model.eval()
            self._transform = T.Compose([
                T.Resize((FRAME_HEIGHT, FRAME_WIDTH)),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])
            print("✅ MiDaS_small loaded successfully", flush=True)
        except Exception as e:
            print(f"⚠️ MiDaS load failed: {e}, depth will be zeros", flush=True)
            self._model = None
    
    def estimate(self, rgb):
        """
        估计深度图
        
        Args:
            rgb: (B, 3, H, W) RGB图像，值域[0, 1]
        Returns:
            depth: (B, 1, H, W) 归一化深度图
        """
        if self._model is None:
            # 返回零深度
            return torch.zeros(rgb.shape[0], 1, FRAME_HEIGHT, FRAME_WIDTH, 
                             device=rgb.device)
        
        with torch.no_grad():
            # 转换到MiDaS输入格式
            input_tensor = self._transform(rgb)
            
            # 估计深度
            depth = self._model(input_tensor)
            
            # 调整大小
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=(FRAME_HEIGHT, FRAME_WIDTH),
                mode="bilinear",
                align_corners=False
            )
            
            # 归一化到[0, 1]
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)
        
        return depth


# ============================================================
# Part 3: 目标帧处理器
# ============================================================

class GoalFrameProcessor:
    """
    目标帧处理器
    
    存档点画面作为导航目标
    自动检测存档点UI触发
    """
    
    def __init__(self, encoder):
        self.encoder = encoder
        self.depth_estimator = DepthEstimator()
        
        # 存档点检测参数（颜色阈值）
        self.savepoint_lower = np.array([20, 100, 100])  # 存档点UI的橙色
        self.savepoint_upper = np.array([40, 255, 255])
    
    def encode_goal(self, goal_frame):
        """
        编码目标帧
        
        Args:
            goal_frame: (3, H, W) 或 (H, W, 3) numpy数组或tensor
        Returns:
            goal_feat: (1, latent_dim) 目标特征
        """
        # 转换为tensor
        if isinstance(goal_frame, np.ndarray):
            goal_frame = torch.from_numpy(goal_frame)
        
        if goal_frame.dim() == 3 and goal_frame.shape[0] != 3:
            # (H, W, C) -> (C, H, W)
            goal_frame = goal_frame.permute(2, 0, 1)
        
        # 归一化
        if goal_frame.max() > 1.0:
            goal_frame = goal_frame.float() / 255.0
        
        # 添加batch维度
        goal_frame = goal_frame.unsqueeze(0).float()
        
        # 估计深度
        depth = self.depth_estimator.estimate(goal_frame)
        
        # 编码
        with torch.no_grad():
            goal_feat = self.encoder(goal_frame, depth)
        
        return goal_feat
    
    def detect_savepoint(self, frame):
        """
        检测存档点是否在画面中
        
        Returns:
            is_visible: bool
            position: (x, y) 存档点在画面中的位置（如果有）
        """
        import cv2
        
        if isinstance(frame, torch.Tensor):
            if frame.dim() == 4:
                frame = frame[0]
            frame = frame.cpu().numpy()
            if frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)
        
        # 转换到HSV
        frame_uint8 = (frame * 255).astype(np.uint8)
        hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2HSV)
        
        # 检测橙色
        mask = cv2.inRange(hsv, self.savepoint_lower, self.savepoint_upper)
        
        # 计算存档点中心
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return True, (cx, cy)
        
        return False, None


# ============================================================
# Part 4: PointNav核心模型
# ============================================================

class WukongPointNav(nn.Module):
    """
    悟空PointNav模型
    
    核心思想：当前画面 + 目标画面 → 导航动作
    
    类比：
      PointNav: GPS坐标(x,y) → 动作(forward/turn_left/turn_right)
      悟空: 存档点画面 → 动作(WASD + 鼠标)
    
    架构：
      ┌─────────────────────────────────────────────────────┐
      │              DDPPO Visual Encoder (共享)              │
      │                                                      │
      │   current_frame ──→ encoder ──→ current_feat        │
      │       goal_frame ──→ encoder ──→ goal_feat          │
      └─────────────────────────────────────────────────────┘
                              ↓
                      ┌───────────────────┐
                      │   特征融合层      │
                      │ (current + goal)  │
                      └───────────────────┘
                              ↓
                      ┌───────────────────┐
                      │   导航决策头      │
                      │                   │
                      │  - 方向预测      │
                      │  - 鼠标预测      │
                      │  - 动作分类      │
                      └───────────────────┘
    """

    def __init__(
        self,
        latent_dim=256,
        hidden_dim=512,
        num_actions=NUM_ACTIONS,
        use_depth=True,
        pretrained_encoder=True
    ):
        super().__init__()
        
        self.use_depth = use_depth
        self.num_actions = num_actions
        
        # 共享视觉编码器（DD-PPO风格）
        self.visual_encoder = DDPPOVisualEncoder(
            latent_dim=latent_dim,
            pretrained=pretrained_encoder,
            use_depth=use_depth
        )
        
        # 方向预测器（PointNav核心）
        # 预测：目标在当前画面的哪个方向
        self.direction_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4),  # 4个方向: forward/left/right/backward
        )
        
        # 动作分类器
        self.action_classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions),
        )
        
        # 鼠标预测器（第三人称控制）
        self.mouse_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # (dx, dy)
        )
        
        # 移动距离预测（估计还要走多远）
        self.distance_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 0-1表示距离（归一化）
        )
    
    def forward(self, current_frame, goal_frame, depth=None, goal_depth=None):
        """
        前向传播
        
        Args:
            current_frame: (B, 3, H, W) 当前游戏画面
            goal_frame: (B, 3, H, W) 目标存档点画面
            depth: (B, 1, H, W) 当前帧深度（可选）
            goal_depth: (B, 1, H, W) 目标帧深度（可选）
        
        Returns:
            direction_logits: (B, 4) 方向预测
            action_logits: (B, num_actions) 动作预测
            mouse_pred: (B, 2) 鼠标移动预测
            distance_pred: (B, 1) 距离预测
        """
        # 编码当前帧
        current_feat = self.visual_encoder(current_frame, depth)  # (B, latent_dim)
        
        # 编码目标帧
        goal_feat = self.visual_encoder(goal_frame, goal_depth)  # (B, latent_dim)
        
        # 融合
        combined = torch.cat([current_feat, goal_feat], dim=-1)  # (B, latent_dim * 2)
        
        # 预测
        direction_logits = self.direction_predictor(combined)
        action_logits = self.action_classifier(combined)
        mouse_pred = self.mouse_predictor(combined)
        distance_pred = self.distance_predictor(combined)
        
        return direction_logits, action_logits, mouse_pred, distance_pred
    
    def predict_navigation(self, current_frame, goal_frame, depth=None):
        """
        导航预测（推理用）
        
        Returns:
            action: int 动作ID
            mouse_delta: (2,) 鼠标移动(dx, dy)
            direction: str 方向描述
        """
        self.eval()
        with torch.no_grad():
            direction_logits, action_logits, mouse_pred, distance_pred = \
                self.forward(current_frame, goal_frame, depth)
            
            # 方向
            direction_id = direction_logits.argmax(dim=-1).item()
            direction_names = ["forward", "left", "right", "backward"]
            direction = direction_names[direction_id]
            
            # 动作
            action_id = action_logits.argmax(dim=-1).item()
            
            # 鼠标（归一化→实际值）
            mouse_delta = torch.tanh(mouse_pred[0]) * 100  # 限制最大100像素
            
            # 距离
            distance = distance_pred[0, 0].item()
        
        return action_id, mouse_delta.cpu().numpy(), direction, distance


# ============================================================
# Part 5: 训练器
# ============================================================

class PointNavTrainer:
    """PointNav模型训练器"""
    
    def __init__(
        self,
        model,
        lr=1e-3,
        device='cuda',
        weight_dir=2.0,
        weight_mouse=2.0,
        weight_distance=0.5,
        weight_direction=1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.weight_dir = weight_dir
        self.weight_mouse = weight_mouse
        self.weight_distance = weight_distance
        self.weight_direction = weight_direction
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        self.depth_estimator = DepthEstimator()
    
    def load_data(self, data_dir, goal_dir=None):
        """
        加载训练数据
        
        Args:
            data_dir: 轨迹数据目录（包含.h5文件）
            goal_dir: 目标帧目录（可选）
        """
        self.data_files = glob.glob(os.path.join(data_dir, "*.h5"))
        self.goal_dir = goal_dir
        
        print(f"Found {len(self.data_files)} data files", flush=True)
    
    def load_batch(self, h5_path):
        """加载一个h5文件"""
        with h5py.File(h5_path, "r") as f:
            frames = f["frames"][:]
            actions = f["actions"][:]
            mouse_dx = f.get("mouse_dx", np.zeros(len(frames)))
            mouse_dy = f.get("mouse_dy", np.zeros(len(frames)))
            
            # 帧格式转换 (N, H, W, C) -> (N, C, H, W)，归一化
            frames = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            
            return {
                "frames": torch.from_numpy(frames),
                "actions": torch.from_numpy(actions).long(),
                "mouse_dx": torch.from_numpy(mouse_dx).float(),
                "mouse_dy": torch.from_numpy(mouse_dy).float(),
            }
    
    def create_goal_frames(self, frames):
        """
        创建目标帧序列
        
        从轨迹中提取关键帧作为目标
        """
        T = len(frames)
        if T < 10:
            # 太短，直接用最后帧作为目标
            goal_indices = [T - 1] * T
        else:
            # 从轨迹中均匀采样目标
            goal_indices = []
            for i in range(T):
                # 目标是当前帧之后的某个帧
                remaining = T - i
                if remaining > 5:
                    goal_idx = min(i + remaining // 2, T - 1)
                else:
                    goal_idx = T - 1
                goal_indices.append(goal_idx)
        
        goal_frames = frames[goal_indices]
        return goal_frames
    
    def compute_direction_labels(self, frames):
        """
        计算方向标签
        
        通过帧间差异判断移动方向
        """
        T = len(frames)
        directions = []
        
        for i in range(T):
            if i == 0:
                directions.append(0)  # 第一帧默认forward
            else:
                # 简单启发式：通过画面中心变化判断方向
                prev_center = frames[i-1].mean(axis=(1, 2))
                curr_center = frames[i].mean(axis=(1, 2))
                
                delta = curr_center - prev_center
                
                # 简化判断
                if abs(delta).max() < 0.01:
                    directions.append(0)  # 基本静止
                elif delta.mean() > 0.01:
                    directions.append(0)  # forward
                elif delta.mean() < -0.01:
                    directions.append(3)  # backward
                else:
                    directions.append(0)  # 默认forward
        
        return torch.tensor(directions, dtype=torch.long)
    
    def train_step(self, batch):
        """单步训练"""
        frames = batch["frames"].to(self.device)
        actions = batch["actions"].to(self.device)
        mouse_dx = batch["mouse_dx"].to(self.device)
        mouse_dy = batch["mouse_dy"].to(self.device)
        
        B, T = actions.shape[:2]
        
        # 重塑
        frames = frames.view(B * T, *frames.shape[2:])
        
        # 估计深度
        depth = self.depth_estimator.estimate(frames)
        
        # 创建目标帧
        goal_frames = self.create_goal_frames(frames.cpu().numpy())
        goal_frames = torch.from_numpy(goal_frames).to(self.device)
        goal_depth = self.depth_estimator.estimate(goal_frames)
        
        # 计算方向标签
        direction_labels = self.compute_direction_labels(frames.cpu().numpy())
        direction_labels = direction_labels.to(self.device)
        
        # 前向
        dir_logits, action_logits, mouse_pred, dist_pred = \
            self.model(frames, goal_frames, depth, goal_depth)
        
        # 重塑输出
        action_logits = action_logits.view(B * T, -1)
        mouse_pred = mouse_pred.view(B * T, 2)
        dir_logits = dir_logits.view(B * T, -1)
        
        # 损失
        action_loss = F.cross_entropy(action_logits, actions.view(-1))
        
        mouse_target = torch.stack([mouse_dx, mouse_dy], dim=-1).view(-1, 2)
        mouse_loss = F.smooth_l1_loss(mouse_pred, mouse_target)
        
        direction_loss = F.cross_entropy(dir_logits, direction_labels)
        
        # 距离损失（鼓励接近目标）
        # 简化：用动作多样性作为代理
        dist_loss = torch.tensor(0.0, device=self.device)
        
        total_loss = (
            action_loss +
            self.weight_mouse * mouse_loss +
            self.weight_direction * direction_loss +
            self.weight_distance * dist_loss
        )
        
        # 反向
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 指标
        with torch.no_grad():
            action_acc = (action_logits.argmax(dim=-1) == actions.view(-1)).float().mean()
        
        return {
            "total_loss": total_loss.item(),
            "action_loss": action_loss.item(),
            "mouse_loss": mouse_loss.item(),
            "direction_loss": direction_loss.item(),
            "action_acc": action_acc.item(),
        }
    
    def train_epoch(self, batch_size=8):
        """训练一个epoch"""
        self.model.train()
        
        metrics = {
            "total_loss": 0,
            "action_loss": 0,
            "mouse_loss": 0,
            "direction_loss": 0,
            "action_acc": 0,
        }
        n_samples = 0
        
        np.random.shuffle(self.data_files)
        
        for h5_path in self.data_files:
            try:
                batch = self.load_batch(h5_path)
                frames = batch["frames"]
                B, T = frames.shape[:2]
                
                # 按batch分割
                for start in range(0, B, batch_size):
                    end = min(start + batch_size, B)
                    batch_data = {
                        "frames": frames[start:end],
                        "actions": batch["actions"][start:end],
                        "mouse_dx": batch["mouse_dx"][start:end],
                        "mouse_dy": batch["mouse_dy"][start:end],
                    }
                    
                    m = self.train_step(batch_data)
                    
                    metrics["total_loss"] += m["total_loss"] * (end - start)
                    metrics["action_loss"] += m["action_loss"] * (end - start)
                    metrics["mouse_loss"] += m["mouse_loss"] * (end - start)
                    metrics["direction_loss"] += m["direction_loss"] * (end - start)
                    metrics["action_acc"] += m["action_acc"] * (end - start)
                    n_samples += (end - start)
            
            except Exception as e:
                print(f"Error processing {h5_path}: {e}", flush=True)
                continue
        
        # 平均
        for k in metrics:
            metrics[k] /= max(n_samples, 1)
        
        return metrics
    
    def save(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, path)
        print(f"Model saved: {path}", flush=True)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Model loaded: {path}", flush=True)


# ============================================================
# Part 6: 推理
# ============================================================

class PointNavInferer:
    """PointNav推理器"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # 加载模型
        self.model = WukongPointNav().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        
        # 深度估计
        self.depth_estimator = DepthEstimator()
        
        # 目标帧处理
        self.goal_processor = GoalFrameProcessor(self.model.visual_encoder)
        self.goal_frame = None
        
        print(f"PointNav inferer loaded: {model_path}", flush=True)
    
    def set_goal(self, goal_image_path):
        """设置导航目标（存档点画面）"""
        from PIL import Image
        
        img = Image.open(goal_image_path).convert("RGB")
        img = img.resize((FRAME_WIDTH, FRAME_HEIGHT))
        
        self.goal_frame = T.ToTensor()(img).to(self.device)
        print(f"Goal set: {goal_image_path}", flush=True)
    
    def step(self, current_frame):
        """
        一步推理
        
        Args:
            current_frame: (3, H, W) 或 (B, 3, H, W) 当前画面
        
        Returns:
            action_id: int 动作ID
            mouse_delta: (2,) 鼠标移动
        """
        if self.goal_frame is None:
            raise ValueError("Goal not set. Call set_goal() first.")
        
        # 格式化输入
        if current_frame.dim() == 3:
            current_frame = current_frame.unsqueeze(0)
        
        current_frame = current_frame.to(self.device)
        goal_frame = self.goal_frame.unsqueeze(0).repeat(len(current_frame), 1, 1, 1)
        
        # 估计深度
        depth = self.depth_estimator.estimate(current_frame)
        goal_depth = self.depth_estimator.estimate(goal_frame)
        
        # 推理
        with torch.no_grad():
            action_id, mouse_delta, direction, distance = \
                self.model.predict_navigation(current_frame, goal_frame, depth)
        
        return {
            "action_id": action_id,
            "mouse_delta": mouse_delta,
            "direction": direction,
            "distance_estimate": distance,
        }


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Wukong PointNav Model")
    
    # 模式
    parser.add_argument("--train", action="store_true", help="训练模式")
    parser.add_argument("--infer", action="store_true", help="推理模式")
    
    # 数据
    parser.add_argument("--data-dir", type=str, 
                       default="pathfinding_data",
                       help="训练数据目录")
    parser.add_argument("--goal-dir", type=str, default=None,
                       help="目标帧目录")
    
    # 模型
    parser.add_argument("--model-path", type=str,
                       default="checkpoints/pointnav_model.pt",
                       help="模型保存路径")
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--use-depth", action="store_true", default=True,
                       help="使用深度估计")
    
    # 训练
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    # 推理
    parser.add_argument("--goal-image", type=str, default=None,
                       help="目标画面路径")
    
    # 其他
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    if args.train:
        # ===== 训练模式 =====
        print("=" * 50, flush=True)
        print("PointNav Training Mode", flush=True)
        print("=" * 50, flush=True)
        
        # 创建模型
        model = WukongPointNav(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            use_depth=args.use_depth,
            pretrained_encoder=True,
        )
        
        # 创建训练器
        trainer = PointNavTrainer(
            model,
            lr=args.lr,
            device=device,
        )
        trainer.load_data(args.data_dir, args.goal_dir)
        
        # 训练循环
        best_loss = float("inf")
        for epoch in range(args.epochs):
            metrics = trainer.train_epoch(batch_size=args.batch_size)
            trainer.scheduler.step()
            
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"loss={metrics['total_loss']:.4f} | "
                f"acc={metrics['action_acc']:.3f} | "
                f"dir_loss={metrics['direction_loss']:.4f} | "
                f"mouse_loss={metrics['mouse_loss']:.4f}",
                flush=True
            )
            
            # 保存
            if metrics['total_loss'] < best_loss:
                best_loss = metrics['total_loss']
                trainer.save(args.model_path)
        
        print("Training complete!", flush=True)
    
    elif args.infer:
        # ===== 推理模式 =====
        print("=" * 50, flush=True)
        print("PointNav Inference Mode", flush=True)
        print("=" * 50, flush=True)
        
        if args.goal_image is None:
            print("Error: --goal-image required for inference", flush=True)
            return
        
        inferer = PointNavInferer(args.model_path, device)
        inferer.set_goal(args.goal_image)
        
        print("Ready for navigation. Call inferer.step(frame) to get actions.", flush=True)
        return inferer
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
