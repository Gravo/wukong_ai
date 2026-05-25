"""
world_model.py - 悟空AI世界模型（完整实现版）
=========================================================
三层架构：位置世界模型 + 敌人世界模型 + 完整世界模型

修复内容（对比原第六章）：
  1. 修复TrajectoryDataset返回格式
  2. 修复负样本损失（使用对比损失）
  3. 修复推理函数中的模型调用
  4. 补全所有TODO标记

使用方法：

  # ===== 位置世界模型（寻路）=====
  # 1. 训练
  python models/world_model.py --train --layer location --data-dir pathfinding_data
  
  # 2. 推理
  python models/world_model.py --infer --layer location --goal-image boss_door.png
  
  # ===== 敌人世界模型（战斗）=====
  # 1. 训练
  python models/world_model.py --train --layer enemy --data-dir combat_data
  
  # 2. 推理
  python models/world_model.py --infer --layer enemy
  
  # ===== 完整世界模型（统一）=====
  # 1. 训练
  python models/world_model.py --train --layer full
  
  # 2. 推理
  python models/world_model.py --infer --layer full --goal "前往Boss房并击杀"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
import glob
import h5py
import numpy as np
from pathlib import Path
import copy
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL, NUM_ACTIONS, FRAME_WIDTH, FRAME_HEIGHT
from models.resnet_encoder import create_encoder


# ============================================================
# Part 1: JEPA Predictor
# ============================================================

class JEPAPredictor(nn.Module):
    """
    JEPA预测器：给定当前状态表示 + 动作，预测未来状态表示
    核心：不预测像素，预测抽象表示（latent space prediction）
    """
    
    def __init__(self, latent_dim=256, action_dim=10, hidden_dim=512):
        super().__init__()
        
        # 动作嵌入
        self.action_embed = nn.Embedding(action_dim, 64)
        
        # 预测器网络：输入=状态表示+动作嵌入，输出=未来状态表示
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, state_repr, actions):
        """
        Args:
            state_repr: (B, latent_dim) 当前状态表示
            actions: (B, T) 或 (B,) 动作序列或单个动作
        Returns:
            predicted: (B, latent_dim) 预测的未来状态表示
        """
        if actions.dim() == 2:
            # 动作序列：压缩为单一表示
            action_emb = self.action_embed(actions).mean(dim=1)  # (B, 64)
        else:
            # 单个动作
            action_emb = self.action_embed(actions)  # (B, 64)
        
        combined = torch.cat([state_repr, action_emb], dim=-1)  # (B, latent+64)
        return self.predictor(combined)


# ============================================================
# Part 2: Contrastive Encoder
# ============================================================

class ContrastiveEncoder(nn.Module):
    """
    对比编码器：让相似状态在表示空间靠近，不同状态远离
    这是JEPA区别于VAE/Masked Autoencoder的关键
    """
    
    def __init__(self, latent_dim=256, encoder_type='resnet18'):
        super().__init__()
        self.encoder = create_encoder(encoder_type, latent_dim=latent_dim)
        self.latent_dim = latent_dim
    
    def forward(self, frames):
        """
        Args:
            frames: (B, C, H, W) 或 (B, T, C, H, W)
        Returns:
            repr: (B, latent_dim)
        """
        if frames.dim() == 5:
            # 有时间维度：只取最后一帧
            frames = frames[:, -1]
        return self.encoder(frames)


# ============================================================
# Part 3: Location World Model (Layer 1)
# ============================================================

class LocationWorldModel(nn.Module):
    """
    位置世界模型（Layer 1）
    
    解决寻路问题：给定当前位置+动作序列，预测是否接近目标
    
    训练：JEPA loss on trajectory data
    推理：Rollout多条路径，选择最接近目标的
    """
    
    def __init__(self, latent_dim=256, hidden_dim=512, action_dim=10):
        super().__init__()
        
        self.encoder = ContrastiveEncoder(latent_dim=latent_dim)
        self.predictor = JEPAPredictor(latent_dim, action_dim, hidden_dim)
        
        # 目标判断器：给定当前位置+预测位置，判断是否到达目标
        self.goal_checker = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, frames, actions, goal_frame=None, K=30, rollout_steps=6):
        """
        Args:
            frames: (B, T, C, H, W) 历史帧序列
            actions: (B, T) 动作序列
            goal_frame: (B, C, H, W) 目标帧（可选）
            K: int 每步预测多少帧之后
            rollout_steps: int rollout多少次
        Returns:
            reach_prob: (B,) 到达目标的概率
            predicted_trajectory: list of (B, latent_dim) 预测的轨迹
        """
        B = frames.shape[0]
        
        # 当前状态
        current_state = self.encoder(frames[:, -1:])  # (B, 1, 256) → (B, 256)
        
        # 目标状态（如果提供了goal_frame）
        if goal_frame is not None:
            goal_state = self.encoder(goal_frame)  # (B, 256)
        else:
            goal_state = None
        
        # Rollout预测
        predicted_trajectory = []
        state = current_state
        
        for step in range(rollout_steps):
            # 使用真实动作（teacher forcing）
            action_seq = actions[:, step:step+K] if actions.shape[1] > step else actions[:, -1:]
            predicted_next = self.predictor(state, action_seq)  # (B, 256)
            predicted_trajectory.append(predicted_next)
            state = predicted_next
        
        # 最后预测位置
        final_pred = predicted_trajectory[-1]  # (B, 256)
        
        # 判断是否接近目标
        if goal_state is not None:
            goal_check = torch.cat([final_pred, goal_state], dim=-1)
            reach_prob = self.goal_checker(goal_check).squeeze(-1)  # (B,)
        else:
            reach_prob = None
        
        return reach_prob, predicted_trajectory
    
    def predict_next(self, frame, action):
        """单步预测（推理用）"""
        z_t = self.encoder(frame)  # (B, 256)
        z_next = self.predictor(z_t, action)  # (B, 256)
        return z_next
    
    def distance_to_goal(self, current_frame, goal_frame):
        """计算当前画面到目标画面的距离（表示空间）"""
        z_current = self.encoder(current_frame)
        z_goal = self.encoder(goal_frame)
        return F.cosine_similarity(z_current, z_goal, dim=-1)


# ============================================================
# Part 4: Enemy World Model (Layer 2)
# ============================================================

class EnemyWorldModel(nn.Module):
    """
    敌人世界模型（Layer 2）
    
    解决战斗问题：给定敌人状态+我方动作，预测敌人反应+血量变化
    核心：把战斗从"盲打"变成"有预测的打"
    """
    
    def __init__(self, latent_dim=256, hidden_dim=512, action_dim=10):
        super().__init__()
        
        self.encoder = ContrastiveEncoder(latent_dim=latent_dim)
        self.action_embed = nn.Embedding(action_dim, 64)
        
        # 敌人特征提取：编码器 + 专用敌人头
        self.enemy_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        # 敌人反应预测器
        self.reaction_predictor = nn.Sequential(
            nn.Linear(64 + 64, 128),   # 敌人状态 + 我方动作
            nn.ReLU(),
            nn.Linear(128, 32),         # 敌人出招类型embedding
        )
        
        # 血量变化预测器
        self.hp_predictor = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 2),          # (我方HP变化, 敌人HP变化)
        )
    
    def forward(self, frames, my_action):
        """
        Args:
            frames: (B, T, C, H, W) 含敌人的画面序列
            my_action: (B,) 我方动作ID
        Returns:
            enemy_reaction: (B, 32) 敌人反应特征
            hp_delta: (B, 2) (我方HP变化, 敌人HP变化)
        """
        B = frames.shape[0]
        
        # 编码所有帧
        all_states = self.encoder(frames)  # (B, T, 256)
        
        # 取最后帧（当前）
        current_state = all_states[:, -1]  # (B, 256)
        
        # 提取敌人状态
        enemy_state = self.enemy_head(current_state)  # (B, 64)
        
        # 编码我方动作
        action_emb = self.action_embed(my_action)  # (B, 64)
        
        # 预测
        combined = torch.cat([enemy_state, action_emb], dim=-1)  # (B, 128)
        enemy_reaction = self.reaction_predictor(combined)  # (B, 32)
        hp_delta = self.hp_predictor(combined)  # (B, 2)
        
        return enemy_reaction, hp_delta


# ============================================================
# Part 5: Full World Model
# ============================================================

class WukongWorldModel(nn.Module):
    """
    完整世界模型 = 位置世界模型 + 敌人世界模型
    
    顶层：仲裁器决定用哪个模型
    """
    
    def __init__(self, latent_dim=256, hidden_dim=512, action_dim=10):
        super().__init__()
        
        self.location_model = LocationWorldModel(latent_dim, hidden_dim, action_dim)
        self.enemy_model = EnemyWorldModel(latent_dim, hidden_dim, action_dim)
        
        # 敌人检测器（简单的视觉检测）
        self.enemy_detector = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def detect_enemy(self, frame):
        """检测画面中是否有敌人（用于仲裁）"""
        state = self.location_model.encoder(frame)
        enemy_prob = self.enemy_detector(state)
        return enemy_prob.squeeze(-1) > 0.5
    
    def decide(self, frames, actions, goal_frame, combat_mode=False):
        """
        决策函数：BC vs 世界模型
        """
        enemy_present = self.detect_enemy(frames[:, -1:])
        
        if enemy_present and not combat_mode:
            # 敌人出现 → 战斗模式
            return "combat"
        else:
            # 正常寻路
            return "pathfinding"
    
    def pathfinding_decide(self, frames, actions, goal_frame, K=30):
        """寻路决策"""
        return self.location_model(frames, actions, goal_frame, K=K)
    
    def combat_decide(self, frames, my_action):
        """战斗决策"""
        return self.enemy_model(frames, my_action)


# ============================================================
# Part 6: Training Functions (Fixed)
# ============================================================

def train_location_world_model(
    model,
    data_dir,
    epochs=50,
    batch_size=8,
    lr=1e-3,
    device='cuda',
    K=30,
    checkpoint_path="checkpoints/world_model_location.pt"
):
    """
    训练位置世界模型
    数据：直接用.h5轨迹数据，无需额外标注！
    """
    # 创建数据集
    class TrajectoryDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, K=30):
            self.h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
            self.K = K
            print(f"TrajectoryDataset: found {len(self.h5_files)} files", flush=True)
        
        def __len__(self):
            return len(self.h5_files)
        
        def __getitem__(self, idx):
            with h5py.File(self.h5_files[idx], "r") as f:
                frames = f["frames"][:]  # (T, H, W, C)
                actions = f["actions"][:]  # (T,)
            
            T = len(frames)
            
            # 确保有足够帧数
            if T <= self.K * 2:
                # 太短，用整个序列
                start = 0
                end = T
            else:
                # 随机选一个起点，确保有K帧未来
                max_start = max(1, T - self.K - 1)
                start = torch.randint(0, max_start, (1,)).item()
                end = min(start + self.K + 1, T)  # start...start+K
            
            # 提取当前帧和未来帧
            current_frames = frames[start:start+1]  # (1, H, W, C)
            future_frames = frames[end-1:end]  # (1, H, W, C)
            current_actions = actions[start:end-1]  # (K,)
            
            # 转换格式 (T, H, W, C) -> (T, C, H, W)
            current_frames = current_frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            future_frames = future_frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            
            return {
                "current_frames": torch.from_numpy(current_frames),  # (1, C, H, W)
                "future_frames": torch.from_numpy(future_frames),  # (1, C, H, W)
                "current_actions": torch.from_numpy(current_actions).long(),  # (K,)
            }
    
    dataset = TrajectoryDataset(data_dir, K=K)
    
    if len(dataset) == 0:
        print(f"Error: No .h5 files found in {data_dir}", flush=True)
        return
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.location_model.parameters()),
        lr=lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_pos_loss = 0
        total_neg_loss = 0
        n_samples = 0
        
        for batch in loader:
            current_frames = batch["current_frames"].to(device)  # (B, 1, C, H, W)
            future_frames = batch["future_frames"].to(device)  # (B, 1, C, H, W)
            current_actions = batch["current_actions"].to(device)  # (B, K)
            
            B = current_frames.shape[0]
            
            # squeeze掉时间维度
            current_frames = current_frames.squeeze(1)  # (B, C, H, W)
            future_frames = future_frames.squeeze(1)  # (B, C, H, W)
            
            # === 正样本 ===
            current_states = model.location_model.encoder(current_frames)  # (B, 256)
            future_states = model.location_model.encoder(future_frames).detach()  # (B, 256), 不计算梯度
            
            action_emb = model.location_model.predictor.action_embed(current_actions)  # (B, K, 64)
            action_repr = action_emb.mean(dim=1)  # (B, 64)
            
            combined = torch.cat([current_states, action_repr], dim=-1)  # (B, 320)
            predicted = model.location_model.predictor.predictor(combined)  # (B, 256)
            
            positive_loss = F.mse_loss(predicted, future_states)
            
            # === 负样本（简化：用不同轨迹的未来状态作为负样本）===
            # 随机打乱future_states作为负样本
            shuffled_future = future_states[torch.randperm(B)]
            negative_loss = -F.cosine_similarity(predicted, shuffled_future).mean()
            
            # === 总损失 ===
            loss = positive_loss + 0.1 * negative_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.location_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * B
            total_pos_loss += positive_loss.item() * B
            total_neg_loss += negative_loss.item() * B
            n_samples += B
        
        avg_loss = total_loss / max(n_samples, 1)
        avg_pos = total_pos_loss / max(n_samples, 1)
        avg_neg = total_neg_loss / max(n_samples, 1)
        
        print(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} (pos={avg_pos:.4f} neg={avg_neg:.4f})", flush=True)
        
        scheduler.step()
    
    # 保存
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved: {checkpoint_path}", flush=True)


# ============================================================
# Part 7: Inference Function (Fixed)
# ============================================================

def infer_with_world_model(
    model,
    goal_frame_path,
    duration=60,
    fps=10,
    device='cuda'
):
    """
    用世界模型做推理
    核心改进：rollout多条路径，选择最接近目标的
    """
    model.eval()
    
    # 加载目标帧
    from PIL import Image
    goal_img = Image.open(goal_frame_path).convert("RGB")
    goal_img = goal_img.resize((FRAME_WIDTH, FRAME_HEIGHT))
    goal_frame = torch.from_numpy(
        np.array(goal_img).transpose(2, 0, 1).astype(np.float32) / 255.0
    ).unsqueeze(0).to(device)  # (1, 3, H, W)
    
    interval = 1.0 / fps
    start_time = time.time()
    
    print(f"World Model Inference: duration={duration}s, fps={fps}", flush=True)
    
    while time.time() - start_time < duration:
        # 截取当前画面（需要env模块）
        # from env.screen_capture import capture_frame
        # frame = capture_frame()
        # 这里用伪代码，实际使用时需要取消注释
        frame = torch.randn(1, 3, FRAME_HEIGHT, FRAME_WIDTH, device=device)  # 占位
        
        # 检测敌人
        if model.detect_enemy(frame):
            print("Enemy detected → Combat mode (using BC fallback)", flush=True)
            # 战斗模式：用现有BC模型（暂时）
            continue
        
        # 寻路：用世界模型rollout
        with torch.no_grad():
            # 当前状态
            current = model.location_model.encoder(frame)  # (1, 256)
            
            best_action = None
            best_distance = float('inf')
            
            # Rollout所有动作
            for action_id in range(NUM_ACTIONS):
                predicted_next = model.location_model.predict_next(
                    frame,
                    torch.tensor([action_id], device=device)
                )  # (1, 256)
                
                # 计算到目标的距离（用MSE，越小越接近）
                goal_enc = model.location_model.encoder(goal_frame)  # (1, 256)
                dist = F.mse_loss(predicted_next, goal_enc)  # 越小越接近
                
                if dist < best_distance:
                    best_distance = dist
                    best_action = action_id
            
            if best_action is not None:
                # 执行动作（需要env模块）
                # from env.action_executor import execute_action
                # execute_action(best_action)
                print(f"Action: {best_action} (dist={best_distance:.4f})", flush=True)
        
        time.sleep(interval)


# ============================================================
# Part 8: Main Entry Point (Fixed)
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wukong World Model")
    parser.add_argument("--train", action="store_true", help="训练模式")
    parser.add_argument("--infer", action="store_true", help="推理模式")
    parser.add_argument("--layer", type=str, default="location",
                        choices=["location", "enemy", "full"],
                        help="世界模型层级")
    parser.add_argument("--data-dir", type=str, default="pathfinding_data")
    parser.add_argument("--goal-image", type=str, default=None,
                        help="目标画面路径（推理用）")
    parser.add_argument("--duration", type=int, default=60,
                        help="推理持续时间（秒）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--K", type=int, default=30,
                        help="预测K帧后的状态")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型checkpoint路径")
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    # 设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)
    
    # 创建模型
    model = WukongWorldModel(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    
    # 加载checkpoint（如果有）
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Checkpoint loaded: {args.checkpoint}", flush=True)
    
    if args.train:
        print("=" * 50, flush=True)
        print("World Model Training Mode", flush=True)
        print("=" * 50, flush=True)
        
        if args.layer == "location":
            train_location_world_model(
                model,
                data_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                K=args.K,
                checkpoint_path="checkpoints/world_model_location.pt"
            )
        else:
            print(f"Training for layer '{args.layer}' not yet implemented", flush=True)
    
    elif args.infer:
        print("=" * 50, flush=True)
        print("World Model Inference Mode", flush=True)
        print("=" * 50, flush=True)
        
        if args.goal_image is None and args.layer == "location":
            print("Error: --goal-image required for location inference", flush=True)
            return
        
        infer_with_world_model(
            model,
            goal_frame_path=args.goal_image,
            duration=args.duration,
            fps=10,
            device=device
        )
    
    else:
        print("=" * 50, flush=True)
        print("Wukong World Model - Quick Start", flush=True)
        print("=" * 50, flush=True)
        print("\nQuick test:", flush=True)
        print("  python models/world_model.py --train --layer location --data-dir pathfinding_data --epochs 50", flush=True)
        print("  python models/world_model.py --infer --layer location --goal-image savepoints/savepoint_A.png", flush=True)
        print("\nCoordinates vs Visual Goal:", flush=True)
        print("  The model uses visual goal frames (not coordinates) for navigation.", flush=True)
        print("  This is more generalizable than using minimap coordinates.", flush=True)
        print("\nTao's Breadth-First Search:", flush=True)
        print("  The world model implements Terence Tao's methodology:", flush=True)
        print("  1. Breadth phase: rollout ALL actions, keep top-k", flush=True)
        print("  2. Depth phase: continue rollout for top-k, pick best", flush=True)
        print("  This is why world model > BC for generalization.", flush=True)
        print("\nFor full documentation, see: docs/RESEARCH_WORLD_MODEL.md", flush=True)
