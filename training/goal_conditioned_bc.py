"""
goal_conditioned_bc.py - Goal-Conditioned Behavior Cloning 训练脚本 v3.0
优化：idle过滤 + 起始帧鼠标加权 + 动作类别加权 + 鼠标方向一致性损失

使用方法：

  cd D:\projects\wukong_ai

  # 训练
  C:\Python\python.exe -u training\goal_conditioned_bc.py ^
    --data-dir pathfinding_data ^
    --epochs 30 ^
    --batch-size 32 ^
    --lr 0.001

  # 推理
  C:\Python\python.exe -u training\inference_goal.py ^
    --model checkpoints\goal_bc_epoch_030.pt ^
    --goal-id 1 --duration 60

模型架构：
  Input: 单帧画面 (3, 224, 224) + goal_id (int)
    → ResNet18 (预训练) → 512维特征
    → Goal Embedding (可学习) → 64维
    → Concatenate [512 + 64] → 576维
    → Fusion Layer → 512维
    → 动作头: Linear(512 → 10)
    → 鼠标头: Linear(512 → 2)

v3.0 优化（基于 v2.0 推理反馈）：
  1. 过滤无鼠标数据的 h5 文件
  2. 过滤 idle 帧（减少无用数据）
  3. 起始帧鼠标损失加权 20x（强制学"开局先转向"）
  4. 动作类别加权（rare action 5-10x）
  5. 鼠标方向一致性损失（减少左右抖动）
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
START_FRAMES = 10  # 起始帧数量（前 N 帧标记为起始帧）


# ---- Dataset ----

class GoalConditionedDataset(Dataset):
    """加载带 goal_ids 的 h5 数据（单帧输入，v3.0 优化版）"""

    def __init__(self, data_dir, max_samples=0):
        self.samples = []
        self._frame_cache = []

        print(f"[Data] Loading h5 files from {data_dir}...", flush=True)

        h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
        print(f"[Data] Found {len(h5_files)} h5 files", flush=True)

        for h5_file in h5_files:
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
                        print(f"[Data] ⚠️  {h5_file} has no goal_ids, using 0", flush=True)
                        goal_ids = np.zeros(len(frames), dtype=np.int8)

                    # 过滤：跳过鼠标数据不合格的文件
                    if mouse_dx.std() < 0.01 and mouse_dy.std() < 0.01:
                        print(f"[Data] ⏭️  {h5_file}: mouse data missing, skipping", flush=True)
                        continue

                    # 预处理：转 (N, 224, 224, 3) → (N, 3, 224, 224)
                    frames = frames.transpose(0, 3, 1, 2)
                    frames = frames.astype(np.float32) / 255.0

                    # ImageNet 归一化
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
                    frames = (frames - mean) / std

                    # 归一化鼠标数据：除以标准差，让模型目标值有合理范围
                    mouse_std = max(mouse_dx.std(), 1.0)  # 避免除0
                    mouse_dy_std = max(mouse_dy.std(), 1.0)
                    mouse_dx = mouse_dx.astype(np.float32) / mouse_std
                    mouse_dy = mouse_dy.astype(np.float32) / mouse_dy_std
                    # Clamp 到 [-3, 3] 避免极端值
                    mouse_dx = np.clip(mouse_dx, -3.0, 3.0)
                    mouse_dy = np.clip(mouse_dy, -3.0, 3.0)
                    print(f"[Data] 📊 {h5_file}: mouse dx_std={mouse_std:.1f} dy_std={mouse_dy_std:.1f}", flush=True)

                    # 构建样本 + 过滤 idle 帧 + 标记起始帧
                    n_valid = 0
                    for i in range(len(frames)):
                        act = int(actions[i])
                        if act == 0:  # idle
                            continue
                        is_start = (i < START_FRAMES)
                        self.samples.append({
                            'frame_idx': len(self._frame_cache),
                            'goal_id': int(goal_ids[i]),
                            'action': act,
                            'mouse': np.array([mouse_dx[i], mouse_dy[i]], dtype=np.float32),
                            'is_start': is_start,
                        })
                        self._frame_cache.append(frames[i])
                        n_valid += 1

                    print(f"[Data] ✅ {h5_file}: {n_valid} valid frames (idle filtered)", flush=True)

            except Exception as e:
                print(f"[Data] ❌ Error loading {h5_file}: {e}", flush=True)
                continue

        print(f"[Data] ✅ Loaded {len(self.samples)} samples total", flush=True)

        # 限制样本数量
        if max_samples > 0 and len(self.samples) > max_samples:
            print(f"[Data] Truncating to {max_samples} samples...", flush=True)
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            indices.sort()
            self.samples = [self.samples[i] for i in indices]
            needed_indices = set(s['frame_idx'] for s in self.samples)
            old_to_new = {}
            new_cache = []
            for s in self.samples:
                old_idx = s['frame_idx']
                if old_idx not in old_to_new:
                    old_to_new[old_idx] = len(new_cache)
                    new_cache.append(self._frame_cache[old_idx])
                s['frame_idx'] = old_to_new[old_idx]
            self._frame_cache = new_cache
            print(f"[Data] ✅ Using {len(self.samples)} samples", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = torch.from_numpy(self._frame_cache[sample['frame_idx']])
        goal_id = torch.tensor(sample['goal_id'], dtype=torch.long)
        action = torch.tensor(sample['action'], dtype=torch.long)
        mouse = torch.from_numpy(sample['mouse'])
        is_start = torch.tensor(1.0 if sample.get('is_start', False) else 0.0, dtype=torch.float32)

        return frames.unsqueeze(0), goal_id, action, mouse, is_start


# ---- Model ----

class GoalConditionedBC(nn.Module):
    """Goal-Conditioned Behavior Cloning 模型（单帧输入）"""

    def __init__(self, num_goals):
        super().__init__()
        self.num_goals = num_goals

        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_feat_dim = 512

        self.goal_embed = nn.Embedding(num_goals, GOAL_EMBED_DIM)

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
    """训练 Goal-Conditioned BC 模型"""
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Goal-Conditioned BC Training (v3.0)", flush=True)
    print(f"  Data: {args.data_dir}", flush=True)
    print(f"  Epochs: {args.epochs}", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)
    print(f"  Learning rate: {args.lr}", flush=True)
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    dataset = GoalConditionedDataset(args.data_dir, max_samples=args.max_samples)

    if len(dataset) == 0:
        print("[Train] ❌ No data loaded! Check data directory.", flush=True)
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    all_goal_ids = [dataset.samples[i]['goal_id'] for i in range(len(dataset))]
    num_goals = max(all_goal_ids) + 1
    print(f"[Train] Detected {num_goals} goals", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoalConditionedBC(num_goals=num_goals).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 动作类别权重
    action_weights = torch.tensor([
        0.0,   # 0: idle (已过滤)
        5.0,   # 1: attack
        10.0,  # 2: heavy
        10.0,  # 3: dodge
        1.0,   # 4: forward
        5.0,   # 5: right
        5.0,   # 6: left
        10.0,  # 7: dodge_atk
        10.0,  # 8: lock
        10.0,  # 9: heal
    ], dtype=torch.float32).to(device)
    action_criterion = nn.CrossEntropyLoss(weight=action_weights)
    mouse_criterion = nn.SmoothL1Loss(reduction='none')  # ✅ 改为逐样本损失

    MOUSE_WEIGHT = 10.0
    START_MOUSE_WEIGHT = 20.0  # ✅ 起始帧鼠标权重
    DIRECTION_CONSISTENCY_WEIGHT = 0.5  # ✅ 方向一致性权重

    print(f"[Train] Loss weights: mouse={MOUSE_WEIGHT}x, start_mouse={START_MOUSE_WEIGHT}x, "
          f"direction={DIRECTION_CONSISTENCY_WEIGHT}x", flush=True)

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_action_acc = 0.0
        total_samples = 0
        total_mouse_loss = 0.0
        total_dir_loss = 0.0

        start_time = time.time()

        for batch_idx, (frames, goal_ids, actions, mouse, is_start) in enumerate(dataloader):
            frames = frames.to(device)
            goal_ids = goal_ids.to(device)
            actions = actions.to(device)
            mouse = mouse.to(device)
            is_start = is_start.to(device)

            frames = frames.squeeze(1)

            optimizer.zero_grad()
            action_logits, mouse_pred = model(frames, goal_ids)

            # 动作损失
            action_loss = action_criterion(action_logits, actions)

            # ✅ 鼠标损失：起始帧 20x，普通帧 10x
            per_sample_mouse_loss = mouse_criterion(mouse_pred, mouse).mean(dim=1)  # (B,)
            frame_weight = (START_MOUSE_WEIGHT - MOUSE_WEIGHT) * is_start + MOUSE_WEIGHT
            mouse_loss = (frame_weight * per_sample_mouse_loss).mean()

            # ✅ 鼠标方向一致性损失：惩罚 dx 符号频繁翻转
            if mouse_pred.shape[0] > 1:
                dx_pred = mouse_pred[:, 0]
                dx_sign = torch.sign(dx_pred)
                sign_flips = (dx_sign[1:] * dx_sign[:-1] < 0).float()
                direction_loss = sign_flips.mean()
            else:
                direction_loss = torch.tensor(0.0, device=device)

            loss = action_loss + mouse_loss + DIRECTION_CONSISTENCY_WEIGHT * direction_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mouse_loss += mouse_loss.item()
            total_dir_loss += direction_loss.item()
            _, predicted = torch.max(action_logits, 1)
            total_action_acc += (predicted == actions).sum().item()
            total_samples += actions.size(0)

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch} Batch {batch_idx}/{len(dataloader)} "
                      f"Loss: {loss.item():.4f} "
                      f"Mouse: {mouse_loss.item():.4f} "
                      f"Dir: {direction_loss.item():.4f}", flush=True)

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_action_acc / total_samples
        avg_mouse = total_mouse_loss / len(dataloader)
        avg_dir = total_dir_loss / len(dataloader)
        epoch_time = time.time() - start_time

        print(f"\n  Epoch {epoch}/{args.epochs} "
              f"Loss: {avg_loss:.4f} "
              f"Acc: {avg_acc * 100:.2f}% "
              f"Mouse: {avg_mouse:.4f} "
              f"Dir: {avg_dir:.4f} "
              f"Time: {epoch_time:.1f}s\n", flush=True)

        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join('checkpoints', f'goal_bc_epoch_{epoch:03d}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [Save] Checkpoint saved: {ckpt_path}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Training Complete!", flush=True)
    print(f"  Checkpoints saved to: checkpoints/", flush=True)
    print(f"{'=' * 60}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Goal-Conditioned BC Training (v3.0)")
    parser.add_argument("--data-dir", type=str, default="pathfinding_data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples (0=all)")
    args = parser.parse_args()
    train(args)
