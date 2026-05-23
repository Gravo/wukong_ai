"""
goal_conditioned_bc_lstm.py - Goal-Conditioned BC with LSTM (Fixed v1.1)
修复：Dataset不再存储帧数据到内存，改为按需加载（on-demand）
序列输入 + LSTM 时序建模，解决单帧信息不足问题
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
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- 配置 ---

ACTION_NAMES = [
    "idle", "attack", "heavy", "dodge", "forward",
    "right", "left", "dodge_atk", "lock", "heal"
]
NUM_ACTIONS = len(ACTION_NAMES)
GOAL_EMBED_DIM = 64
DEFAULT_SEQ_LEN = 10  # 默认序列长度 (0.67秒 @15fps)


# ---- Dataset (Fixed: On-Demand Loading) ---

class GoalConditionedDatasetLSTM(Dataset):
    """
    加载带 goal_ids 的 h5 数据（LSTM版本：序列输入，按需加载帧）
    每个样本是一个序列：(T, 3, 224, 224) + (T,) goal_ids → 预测最后一帧的动作
    优化：不存储帧数据在内存，只在 __getitem__ 时从 h5 读取
    """

    def __init__(self, data_dir, seq_length=DEFAULT_SEQ_LEN):
        """
        Args:
            data_dir: h5 文件目录
            seq_length: 序列长度（帧数）
        """
        self.seq_length = seq_length
        self.samples = []  # 每个元素：(file_path, start_idx, goal_ids, target_action, target_mouse)

        print(f"[Data] Loading h5 files from {data_dir} (LSTM version, seq_len={seq_length}, on-demand)...")

        h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        print(f"[Data] Found {len(h5_files)} h5 files")

        for h5_file in h5_files:
            h5_path = os.path.join(data_dir, h5_file)
            try:
                with h5py.File(h5_path, 'r') as f:
                    if 'actions' not in f:
                        print(f"[Data] ❌ {h5_file} has no actions, skipping")
                        continue

                    n_frames = len(f['frames'])  # 只读取元数据，不加载帧数据
                    actions = f['actions'][:]
                    mouse_dx = f['mouse_dx'][:]
                    mouse_dy = f['mouse_dy'][:]

                    # 加载 goal_ids（如果不存在则全部设为 0）
                    if 'goal_ids' in f:
                        goal_ids = f['goal_ids'][:]
                    else:
                        print(f"[Data] ⚠️  {h5_file} has no goal_ids, using 0")
                        goal_ids = np.zeros(n_frames, dtype=np.int8)

                    # 归一化鼠标数据到 [-1, 1]
                    mouse_dx_norm = mouse_dx.astype(np.float32) / 224.0
                    mouse_dy_norm = mouse_dy.astype(np.float32) / 224.0

                    # 构建序列样本（只存储索引，不存储帧数据）
                    if n_frames < seq_length:
                        print(f"[Data] ⚠️  {h5_file} has only {n_frames} frames (< {seq_length}), skipping")
                        continue

                    for start_idx in range(0, n_frames - seq_length + 1, seq_length // 2):  # 50% 重叠
                        end_idx = start_idx + seq_length
                        seq_goal_ids = goal_ids[start_idx:end_idx]  # (T,) - 很小，可以存储

                        # 目标：预测最后一帧的动作和鼠标
                        target_action = int(actions[start_idx + seq_length - 1])
                        target_mouse = np.array([mouse_dx_norm[start_idx + seq_length - 1], mouse_dy_norm[start_idx + seq_length - 1]], dtype=np.float32)

                        self.samples.append({
                            'file_path': h5_path,
                            'start_idx': start_idx,
                            'goal_ids': seq_goal_ids,  # (T,)
                            'target_action': target_action,
                            'target_mouse': target_mouse,  # (2,)
                        })

            except Exception as e:
                print(f"[Data] ❌ Error loading {h5_file}: {e}")
                continue

        print(f"[Data] ✅ Loaded {len(self.samples)} sequences (seq_len={seq_length}, on-demand)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_path = sample['file_path']
        start_idx = sample['start_idx']
        goal_ids = torch.tensor(sample['goal_ids'], dtype=torch.long)  # (T,)
        target_action = torch.tensor(sample['target_action'], dtype=torch.long)  # scalar
        target_mouse = torch.from_numpy(sample['target_mouse'])  # (2,)

        # 从 h5 文件按需加载帧数据
        with h5py.File(file_path, 'r') as f:
            frames = f['frames'][start_idx:start_idx + self.seq_length]  # (T, 224, 224, 3)
            
            # 预处理：转 (T, 224, 224, 3) → (T, 3, 224, 224)
            frames = frames.transpose(0, 3, 1, 2)  # (T, 3, 224, 224)
            frames = frames.astype(np.float32) / 255.0

            # ImageNet 归一化
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
            frames = (frames - mean) / std

        frames = torch.from_numpy(frames)  # (T, 3, 224, 224)

        return frames, goal_ids, target_action, target_mouse


# ---- Model ----

class GoalConditionedBC_LSTM(nn.Module):
    """Goal-Conditioned BC 模型（LSTM版本：时序建模）"""

    def __init__(self, num_goals, seq_length=DEFAULT_SEQ_LEN, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.num_goals = num_goals
        self.seq_length = seq_length
        self.lstm_hidden = lstm_hidden

        # 视觉编码器（ResNet18）
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])  # 去掉分类头
        self.visual_feat_dim = 512

        # Goal Embedding
        self.goal_embed = nn.Embedding(num_goals, GOAL_EMBED_DIM)

        # LSTM 时序建模
        input_dim = self.visual_feat_dim + GOAL_EMBED_DIM  # 512 + 64 = 576
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.5 if lstm_layers > 1 else 0.0
        )

        # 输出头
        self.action_head = nn.Linear(lstm_hidden, NUM_ACTIONS)
        self.mouse_head = nn.Linear(lstm_hidden, 2)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, frames, goal_ids):
        """
        Args:
            frames: (B, T, 3, 224, 224)  # 序列帧
            goal_ids: (B, T)  # 每个时间步一个 goal_id
        Returns:
            action_logits: (B, NUM_ACTIONS)
            mouse_pred: (B, 2)
        """
        batch_size, seq_len = frames.shape[0], frames.shape[1]

        # 视觉特征提取（每帧独立）
        frames_flat = frames.view(batch_size * seq_len, 3, 224, 224)  # (B*T, 3, 224, 224)
        visual_feat = self.visual_encoder(frames_flat)  # (B*T, 512, 1, 1)
        visual_feat = visual_feat.view(visual_feat.size(0), -1)  # (B*T, 512)
        visual_feat = visual_feat.view(batch_size, seq_len, -1)  # (B, T, 512)

        # Goal embedding（每帧独立）
        goal_emb = self.goal_embed(goal_ids)  # (B, T, 64)

        # 融合
        fused = torch.cat([visual_feat, goal_emb], dim=-1)  # (B, T, 576)

        # LSTM 时序建模
        lstm_out, (h_n, c_n) = self.lstm(fused)  # lstm_out: (B, T, hidden)

        # 使用最后一个时间步的隐藏状态
        last_hidden = lstm_out[:, -1, :]  # (B, hidden)
        feat = self.dropout(last_hidden)  # (B, hidden)

        action_logits = self.action_head(feat)  # (B, NUM_ACTIONS)
        mouse_pred = self.mouse_head(feat)  # (B, 2)

        return action_logits, mouse_pred


# ---- Training ----

def train(args):
    """训练 Goal-Conditioned BC 模型（LSTM版本）"""
    print(f"\n{'=' * 60}")
    print(f"  Goal-Conditioned BC Training (LSTM v1.1 - Fixed)")
    print(f"  Data: {args.data_dir}")
    print(f"  Sequence Length: {args.seq_length}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  LSTM Hidden: 128")
    print(f"{'=' * 60}\n")

    # 数据集
    dataset = GoalConditionedDatasetLSTM(args.data_dir, seq_length=args.seq_length)

    if len(dataset) == 0:
        print("[Train] ❌ No data loaded! Check data directory.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoalConditionedBC_LSTM(num_goals=1, seq_length=args.seq_length).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 损失函数
    action_criterion = nn.CrossEntropyLoss()
    mouse_criterion = nn.SmoothL1Loss()

    # 训练循环
    print(f"[Train] Starting training...")
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_action_acc = 0.0
        total_samples = 0

        start_time = time.time()

        for batch_idx, (frames, goal_ids, target_actions, target_mouse) in enumerate(dataloader):
            frames = frames.to(device)  # (B, T, 3, 224, 224)
            goal_ids = goal_ids.to(device)  # (B, T)
            target_actions = target_actions.to(device)  # (B,)
            target_mouse = target_mouse.to(device)  # (B, 2)

            optimizer.zero_grad()

            action_logits, mouse_pred = model(frames, goal_ids)

            # 损失
            action_loss = action_criterion(action_logits, target_actions)
            mouse_loss = mouse_criterion(mouse_pred, target_mouse)
            loss = action_loss + 2.0 * mouse_loss  # 鼠标损失权重 2.0x

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(action_logits, 1)
            total_action_acc += (predicted == target_actions).sum().item()
            total_samples += target_actions.size(0)

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch} Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")

        scheduler.step()

        # Epoch 统计
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_action_acc / total_samples
        epoch_time = time.time() - start_time

        print(f"\n  Epoch {epoch}/{args.epochs} "
              f"Loss: {avg_loss:.4f} "
              f"Acc: {avg_acc * 100:.2f}% "
              f"Time: {epoch_time:.1f}s\n")

        # 保存检查点
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join('checkpoints', f'goal_bc_lstm_epoch_{epoch:03d}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [Save] Checkpoint saved: {ckpt_path}")

    print(f"\n{'=' * 60}")
    print(f"  Training Complete!")
    print(f"  Checkpoints saved to: checkpoints/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="pathfinding_data",
                        help="Directory containing h5 files")
    parser.add_argument("--seq-length", type=int, default=10,
                        help="Sequence length (frames)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    train(args)