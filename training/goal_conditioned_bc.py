"""
goal_conditioned_bc.py - Goal-Conditioned Behavior Cloning 训练脚本 v2.0
简化设计：单帧输入 + Goal Embedding（先确保能跑通）

使用方法：

  cd D:\projects\wukong_ai
  
  # 1. 用 v3.0 脚本录制带 goal 标注的数据
  C:\Python\python.exe -u training/data_collector_v3.py ^
    --duration 600 --fps 15 ^
    --goals-file goals.txt
  
  # 2. 训练 Goal-Conditioned BC 模型（简化版，单帧输入）
  C:\Python\python.exe -u training\goal_conditioned_bc.py ^
    --data-dir pathfinding_data ^
    --epochs 50 ^
    --batch-size 32 ^
    --lr 0.001
  
  # 3. 推理测试（待实现）
  C:\Python\python.exe -u training\inference_goal.py ^
    --model checkpoints\goal_bc_epoch_050.pt ^
    --duration 60 ^
    --fps 10

模型架构（简化版）：
  Input: 单帧画面 (3, 224, 224) + goal_id (int)
    ↓
  ResNet18 (预训练) → 512维特征
    ↓
  Goal Embedding (可学习) → 64维
    ↓
  Concatenate [512 + 64] → 576维
    ↓
  Fusion Layer: Linear(576 → 512) + ReLU + Dropout
    ↓
  动作头: Linear(512 → 10) + Softmax
  鼠标头: Linear(512 → 2) + SmoothL1Loss
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py

# ---- 配置 ----

ACTION_NAMES = [
    "idle", "attack", "heavy", "dodge", "forward",
    "right", "left", "dodge_atk", "lock", "heal"
]
NUM_ACTIONS = len(ACTION_NAMES)
GOAL_EMBED_DIM = 64


# ---- Dataset ----

class GoalConditionedDataset(Dataset):
    """加载带 goal_ids 的 h5 数据（简化版：单帧输入）"""

    def __init__(self, data_dir):
        """
        Args:
            data_dir: h5 文件目录
        """
        self.samples = []
        self.goal_names = []  # 可选：保存 goal 名称

        print(f"[Data] Loading h5 files from {data_dir}...", flush=True)

        h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        print(f"[Data] Found {len(h5_files)} h5 files", flush=True)

        for h5_file in h5_files:
            h5_path = os.path.join(data_dir, h5_file)
            try:
                with h5py.File(h5_path, 'r') as f:
                    frames = f['frames'][:]  # (N, 224, 224, 3)
                    actions = f['actions'][:]  # (N,)
                    mouse_dx = f['mouse_dx'][:]
                    mouse_dy = f['mouse_dy'][:]

                    # ✅ 加载 goal_ids（如果不存在则全部设为 0）
                    if 'goal_ids' in f:
                        goal_ids = f['goal_ids'][:]
                    else:
                        print(f"[Data] ⚠️  {h5_file} has no goal_ids, using 0", flush=True)
                        goal_ids = np.zeros(len(frames), dtype=np.int8)

                    # 预处理：转 (N, 224, 224, 3) → (N, 3, 224, 224)
                    # PyTorch expects (N, C, H, W)
                    frames = frames.transpose(0, 3, 1, 2)  # (N, 3, 224, 224)
                    frames = frames.astype(np.float32) / 255.0

                    # ImageNet 归一化（注意 dtype=float32）
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
                    frames = (frames - mean) / std

                    # 构建样本（单帧输入）
                    for i in range(len(frames)):
                        self.samples.append({
                            'frames': frames[i],  # (3, 224, 224)
                            'goal_id': int(goal_ids[i]),
                            'action': int(actions[i]),
                            'mouse': np.array([mouse_dx[i], mouse_dy[i]], dtype=np.float32),
                        })

            except Exception as e:
                print(f"[Data] ❌ Error loading {h5_file}: {e}", flush=True)
                continue

        print(f"[Data] ✅ Loaded {len(self.samples)} samples", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = torch.from_numpy(sample['frames'])  # (3, 224, 224)
        goal_id = torch.tensor(sample['goal_id'], dtype=torch.long)  # scalar
        action = torch.tensor(sample['action'], dtype=torch.long)  # scalar
        mouse = torch.from_numpy(sample['mouse'])  # (2,)

        return frames.unsqueeze(0), goal_id, action, mouse
        # frames: (1, 3, 224, 224)  # 添加 batch 维度


# ---- Model ----

class GoalConditionedBC(nn.Module):
    """Goal-Conditioned Behavior Cloning 模型（简化版：单帧输入）"""

    def __init__(self, num_goals):
        super().__init__()
        self.num_goals = num_goals

        # 视觉编码器（ResNet18）
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])  # 去掉分类头
        self.visual_feat_dim = 512

        # Goal Embedding
        self.goal_embed = nn.Embedding(num_goals, GOAL_EMBED_DIM)

        # 融合层
        fused_dim = self.visual_feat_dim + GOAL_EMBED_DIM  # 512 + 64 = 576
        self.fusion = nn.Linear(fused_dim, 512)
        self.action_head = nn.Linear(512, NUM_ACTIONS)
        self.mouse_head = nn.Linear(512, 2)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, frames, goal_ids):
        """
        Args:
            frames: (B, 3, 224, 224)  # 单帧，无序列维度
            goal_ids: (B,)  # 每个样本一个 goal_id
        Returns:
            action_logits: (B, NUM_ACTIONS)
            mouse_pred: (B, 2)
        """
        # 视觉特征提取
        visual_feat = self.visual_encoder(frames)  # (B, 512, 1, 1)
        visual_feat = visual_feat.view(visual_feat.size(0), -1)  # (B, 512)

        # Goal embedding
        goal_emb = self.goal_embed(goal_ids)  # (B, 64)

        # 融合
        fused = torch.cat([visual_feat, goal_emb], dim=-1)  # (B, 576)
        feat = self.relu(self.fusion(fused))  # (B, 512)
        feat = self.dropout(feat)

        action_logits = self.action_head(feat)  # (B, NUM_ACTIONS)
        mouse_pred = self.mouse_head(feat)  # (B, 2)

        return action_logits, mouse_pred


# ---- Training ----

def train(args):
    """训练 Goal-Conditioned BC 模型"""
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Goal-Conditioned Behavior Cloning Training (v2.0)", flush=True)
    print(f"  Data: {args.data_dir}", flush=True)
    print(f"  Epochs: {args.epochs}", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)
    print(f"  Learning rate: {args.lr}", flush=True)
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    # 数据集
    dataset = GoalConditionedDataset(args.data_dir)

    if len(dataset) == 0:
        print("[Train] ❌ No data loaded! Check data directory.", flush=True)
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 估算 goal 数量（从数据中）
    all_goal_ids = [dataset.samples[i]['goal_id'] for i in range(len(dataset))]
    num_goals = max(all_goal_ids) + 1
    print(f"[Train] Detected {num_goals} goals", flush=True)

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoalConditionedBC(num_goals=num_goals).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 损失函数
    action_criterion = nn.CrossEntropyLoss()
    mouse_criterion = nn.SmoothL1Loss()

    # 训练循环
    print(f"[Train] Starting training...", flush=True)
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_action_acc = 0.0
        total_samples = 0

        start_time = time.time()

        for batch_idx, (frames, goal_ids, actions, mouse) in enumerate(dataloader):
            frames = frames.to(device)
            goal_ids = goal_ids.to(device)
            actions = actions.to(device)
            mouse = mouse.to(device)

            # ⚠️ 重要：去掉 frames 的第1维度（DataLoader 添加的 batch 维度 + __getitem__ 添加的序列维度）
            # frames: (B, 1, 3, 224, 224) → (B, 3, 224, 224)
            frames = frames.squeeze(1)

            optimizer.zero_grad()

            action_logits, mouse_pred = model(frames, goal_ids)

            # 损失
            action_loss = action_criterion(action_logits, actions)
            mouse_loss = mouse_criterion(mouse_pred, mouse)
            loss = action_loss + 2.0 * mouse_loss  # 鼠标损失权重 2.0x

            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(action_logits, 1)
            total_action_acc += (predicted == actions).sum().item()
            total_samples += actions.size(0)

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch} Batch {batch_idx}/{len(dataloader)} "
                      f"Loss: {loss.item():.4f}", flush=True)

        scheduler.step()

        # Epoch 统计
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_action_acc / total_samples
        epoch_time = time.time() - start_time

        print(f"\n  Epoch {epoch}/{args.epochs} "
              f"Loss: {avg_loss:.4f} "
              f"Acc: {avg_acc * 100:.2f}% "
              f"Time: {epoch_time:.1f}s\n", flush=True)

        # 保存检查点
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join('checkpoints', f'goal_bc_epoch_{epoch:03d}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [Save] Checkpoint saved: {ckpt_path}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Training Complete!", flush=True)
    print(f"  Checkpoints saved to: checkpoints/", flush=True)
    print(f"{'=' * 60}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Goal-Conditioned Behavior Cloning Training (v2.0)")

    parser.add_argument("--data-dir", type=str, default="pathfinding_data",
                        help="Directory containing h5 files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    train(args)
