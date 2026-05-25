#!/usr/bin/env python3
"""
快速训练脚本 - 用小数据集跑通 Goal-Conditioned BC
支持 --max-samples 参数限制样本数
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

import h5py

# ---- 配置 ---

ACTION_NAMES = [
    "idle", "attack", "heavy", "dodge", "forward",
    "right", "left", "dodge_atk", "lock", "heal"
]
NUM_ACTIONS = len(ACTION_NAMES)
GOAL_EMBED_DIM = 64


# ---- Dataset ----

class SmallGoalDataset(Dataset):
    """加载带 goal_ids 的 h5 数据（限制样本数）"""

    def __init__(self, data_dir, max_samples=None):
        self.samples = []
        
        print(f"[Data] Loading h5 files from {data_dir}...", flush=True)
        
        h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        print(f"[Data] Found {len(h5_files)} h5 files", flush=True)
        
        total_loaded = 0
        
        for h5_file in h5_files:
            if max_samples is not None and total_loaded >= max_samples:
                break
            
            h5_path = os.path.join(data_dir, h5_file)
            try:
                with h5py.File(h5_path, 'r') as f:
                    frames = f['frames'][:]
                    actions = f['actions'][:]
                    mouse_dx = f['mouse_dx'][:]
                    mouse_dy = f['mouse_dy'][:]
                    
                    if 'goal_ids' in f:
                        goal_ids = f['goal_ids'][:]
                    else:
                        goal_ids = np.zeros(len(frames), dtype=np.int8)
                    
                    # 预处理
                    frames = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
                    frames = (frames - mean) / std
                    
                    mouse_dx = mouse_dx.astype(np.float32) / 224.0
                    mouse_dy = mouse_dy.astype(np.float32) / 224.0
                    
                    # 添加样本（限制数量）
                    for i in range(len(frames)):
                        if max_samples is not None and total_loaded >= max_samples:
                            break
                        
                        self.samples.append({
                            'frames': frames[i],
                            'goal_id': int(goal_ids[i]),
                            'action': int(actions[i]),
                            'mouse': np.array([mouse_dx[i], mouse_dy[i]], dtype=np.float32),
                        })
                        total_loaded += 1
            
            except Exception as e:
                print(f"[Data] ❌ Error loading {h5_file}: {e}", flush=True)
                continue
        
        print(f"[Data] ✅ Loaded {len(self.samples)} samples", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = torch.from_numpy(sample['frames'])
        goal_id = torch.tensor(sample['goal_id'], dtype=torch.long)
        action = torch.tensor(sample['action'], dtype=torch.long)
        mouse = torch.from_numpy(sample['mouse'])
        
        return frames.unsqueeze(0), goal_id, action, mouse


# ---- Model ----

class GoalConditionedBC(nn.Module):
    """Goal-Conditioned BC 模型（简化版：单帧输入）"""

    def __init__(self, num_goals):
        super().__init__()
        self.num_goals = num_goals
        
        # 视觉编码器
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_feat_dim = 512
        
        # Goal Embedding
        self.goal_embed = nn.Embedding(num_goals, GOAL_EMBED_DIM)
        
        # 融合层
        fused_dim = self.visual_feat_dim + GOAL_EMBED_DIM
        self.fusion = nn.Linear(fused_dim, 512)
        self.action_head = nn.Linear(512, NUM_ACTIONS)
        self.mouse_head = nn.Linear(512, 2)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, frames, goal_ids):
        visual_feat = self.visual_encoder(frames)
        visual_feat = visual_feat.view(visual_feat.size(0), -1)
        
        goal_emb = self.goal_embed(goal_ids)
        
        fused = torch.cat([visual_feat, goal_emb], dim=-1)
        feat = self.relu(self.fusion(fused))
        feat = self.dropout(feat)
        
        action_logits = self.action_head(feat)
        mouse_pred = self.mouse_head(feat)
        
        return action_logits, mouse_pred


# ---- Training ----

def train(args):
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Quick Training (Small Dataset)", flush=True)
    print(f"  Data: {args.data_dir}", flush=True)
    print(f"  Max samples: {args.max_samples}", flush=True)
    print(f"  Epochs: {args.epochs}", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}", flush=True)
    print(f"{'=' * 60}\n", flush=True)
    
    # 数据集
    dataset = SmallGoalDataset(args.data_dir, max_samples=args.max_samples)
    
    if len(dataset) == 0:
        print("[Train] ❌ No data loaded!", flush=True)
        return
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_goals = max([dataset.samples[i]['goal_id'] for i in range(len(dataset))]) + 1
    print(f"[Train] Detected {num_goals} goals", flush=True)
    
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
            
            # 去掉额外的维度
            frames = frames.squeeze(1)
            
            optimizer.zero_grad()
            
            action_logits, mouse_pred = model(frames, goal_ids)
            
            action_loss = action_criterion(action_logits, actions)
            mouse_loss = mouse_criterion(mouse_pred, mouse)
            loss = action_loss + 2.0 * mouse_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(action_logits, 1)
            total_action_acc += (predicted == actions).sum().item()
            total_samples += actions.size(0)
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch} Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}", flush=True)
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_action_acc / total_samples
        epoch_time = time.time() - start_time
        
        print(f"\n  Epoch {epoch}/{args.epochs} Loss: {avg_loss:.4f} Acc: {avg_acc:.2%} Time: {epoch_time:.1f}s\n", flush=True)
        
        # 保存检查点
        if epoch % 5 == 0 or epoch == args.epochs:
            checkpoint_path = f"checkpoints/goal_bc_quick_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [Save] Checkpoint saved: {checkpoint_path}", flush=True)
    
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Training Complete!", flush=True)
    print(f"  Checkpoints saved to: checkpoints/", flush=True)
    print(f"{'=' * 60}\n", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick Goal-Conditioned BC Training')
    parser.add_argument('--data-dir', type=str, default='pathfinding_data', help='h5 data directory')
    parser.add_argument('--max-samples', type=int, default=2000, help='Maximum number of samples to load')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    train(args)
