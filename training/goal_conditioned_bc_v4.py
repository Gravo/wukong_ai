"""
goal_conditioned_bc_v4.py - 修复版训练脚本 v4.1

核心修复（v4.0 → v4.1）：
1. 动作编码正确：
   - action 4 = forward（前进）→ 只学鼠标回归，不学动作分类
   - action 5 = right（右转）→ 动作分类 + 鼠标回归
   - action 6 = left（左转）→ 动作分类 + 鼠标回归
2. 有效样本：有鼠标移动(dx>1) OR action ∈ {5,6}
3. 过滤掉纯idle/forward无鼠标的帧
4. 鼠标回归用连续MSE，不做离散分类

使用方法：
  cd D:\projects\wukong_ai
  C:\Python\python.exe -u training/goal_conditioned_bc_v4.py ^
    --data-dir pathfinding_data ^
    --epochs 30 ^
    --batch-size 4 ^
    --lr 0.001
"""

import os, sys, time, argparse, h5py, glob, json, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.goal_conditioned_bc import GoalConditionedBC


class EffectiveDataset(Dataset):
    """修复后的数据集，只包含有效样本"""

    def __init__(self, data_dir, max_samples=0):
        self.samples = []  # [(h5_path, frame_idx, action, mouse_dx, mouse_dy, goal_id, is_turn_action)]
        self.mouse_dx_all = []
        self.mouse_dy_all = []

        print(f"[Data] Loading from {data_dir}...")

        for h5_file in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
            self._load_file(h5_file)

        print(f"[Data] Loaded {len(self.samples)} effective samples")

        # 计算全局鼠标标准差
        self.mouse_dx_all = np.array(self.mouse_dx_all, dtype=np.float32)
        self.mouse_dy_all = np.array(self.mouse_dy_all, dtype=np.float32)
        self.mouse_dx_std = max(self.mouse_dx_all.std(), 1.0)
        self.mouse_dy_std = max(self.mouse_dy_all.std(), 1.0)
        del self.mouse_dx_all, self.mouse_dy_all

        print(f"[Data] Mouse std: dx={self.mouse_dx_std:.1f}, dy={self.mouse_dy_std:.1f}")

        # 样本统计
        turn_action_count = sum(1 for s in self.samples if s[2] in [5, 6])
        mouse_count = sum(1 for s in self.samples if abs(s[3]) > 1.0)
        print(f"[Data] Turn action frames (5/6): {turn_action_count}")
        print(f"[Data] Mouse frames: {mouse_count}")

        # 限制样本数量
        if max_samples > 0 and len(self.samples) > max_samples:
            print(f"[Data] Truncating to {max_samples}...")
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

    def _load_file(self, h5_path):
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'frames' not in f:
                    return

                frames = f['frames'][:]
                actions = f['actions'][:]
                mouse_dx = f['mouse_dx'][:]
                mouse_dy = f['mouse_dy'][:]
                goal_ids = f['goal_ids'][:] if 'goal_ids' in f else np.zeros(len(frames), dtype=np.int8)

                n = len(frames)
                for i in range(n):
                    act = int(actions[i])
                    has_mouse = abs(mouse_dx[i]) > 1.0 or abs(mouse_dy[i]) > 1.0
                    is_turn_action = act in [5, 6]  # right/left

                    # 有效样本：有鼠标移动 OR 有转向动作
                    if not (has_mouse or is_turn_action):
                        continue

                    self.samples.append((
                        h5_path, i, act,
                        float(mouse_dx[i]), float(mouse_dy[i]),
                        int(goal_ids[i]), is_turn_action
                    ))
                    self.mouse_dx_all.append(float(mouse_dx[i]))
                    self.mouse_dy_all.append(float(mouse_dy[i]))

        except Exception as e:
            print(f"[Data] Error loading {h5_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_path, fidx, action, mdx, mdy, goal_id, is_turn_action = self.samples[idx]

        with h5py.File(h5_path, 'r') as f:
            frame = f['frames'][fidx]

        # 预处理
        if frame.shape[0] != 224 or frame.shape[1] != 224:
            frame = cv2.resize(frame, (224, 224))

        frame = frame.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        frame = (frame - mean) / std
        frame = frame.transpose(2, 0, 1)

        # 归一化鼠标（用标准差缩放，让目标值在合理范围）
        mdx = np.clip(mdx / self.mouse_dx_std, -3.0, 3.0)
        mdy = np.clip(mdy / self.mouse_dy_std, -3.0, 3.0)

        # action: forward=0, right=1, left=2（重新映射，方便分类）
        if action == 4:
            action_label = 0  # forward
        elif action == 5:
            action_label = 1  # right
        elif action == 6:
            action_label = 2  # left
        else:
            action_label = 0

        return (
            torch.from_numpy(frame),
            torch.tensor(action_label, dtype=torch.long),  # 0=forward, 1=right, 2=left
            torch.tensor([mdx, mdy], dtype=torch.float32),
            torch.tensor(goal_id, dtype=torch.long),
            torch.tensor(is_turn_action, dtype=torch.bool)  # True=right/left, False=forward
        )


def train_epoch(model, dataloader, optimizer, device, epoch,
               mouse_weight=10.0, turn_action_weight=5.0, direction_weight=0.5):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_action_loss = 0
    total_mouse_loss = 0
    correct = 0
    total = 0

    for batch_idx, (frames, action_labels, mouse_target, goal_ids, is_turn_action) in enumerate(dataloader):
        frames = frames.to(device)
        action_labels = action_labels.to(device)
        mouse_target = mouse_target.to(device)
        goal_ids = goal_ids.to(device)
        is_turn_action = is_turn_action.to(device)

        optimizer.zero_grad()

        action_logits, mouse_pred = model(frames, goal_ids)

        # ===== 1. 动作分类损失（只对转向动作帧 action=1/2，即right/left）=====
        turn_mask = is_turn_action.float()  # [batch]
        if turn_mask.sum() > 0:
            action_loss = F.cross_entropy(action_logits, action_labels, reduction='none')
            action_loss = (action_loss * turn_mask).sum() / max(turn_mask.sum(), 1.0)
        else:
            action_loss = torch.tensor(0.0, device=device)

        # ===== 2. 鼠标回归损失（所有样本）=====
        mouse_criterion = nn.MSELoss(reduction='none')
        mouse_loss_raw = mouse_criterion(mouse_pred, mouse_target)  # [batch, 2]
        mouse_loss = mouse_loss_raw.mean()

        # ===== 3. 方向一致性损失（减少抖动）=====
        if mouse_pred.shape[0] > 1:
            direction_loss = -torch.mean(
                F.cosine_similarity(mouse_pred[:-1], mouse_pred[1:], dim=1)
            )
        else:
            direction_loss = torch.tensor(0.0, device=device)

        # ===== 4. 总损失 =====
        loss = action_loss + mouse_weight * mouse_loss + direction_weight * direction_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_action_loss += action_loss.item()
        total_mouse_loss += mouse_loss.item()

        # 准确率（只统计转向动作帧）
        _, predicted = torch.max(action_logits, 1)
        turn_correct = (predicted == action_labels).float() * turn_mask
        correct += turn_correct.sum().item()
        total += turn_mask.sum().item()

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} Batch {batch_idx}/{len(dataloader)} "
                  f"Loss: {loss.item():.4f} Action: {action_loss.item():.4f} "
                  f"Mouse: {mouse_loss.item():.4f}", flush=True)

    avg_loss = total_loss / max(len(dataloader), 1)
    avg_action_loss = total_action_loss / max(len(dataloader), 1)
    avg_mouse_loss = total_mouse_loss / max(len(dataloader), 1)
    acc = 100.0 * correct / max(total, 1)

    return avg_loss, acc, avg_action_loss, avg_mouse_loss


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Goal-Conditioned BC v4.1 (Fixed)")
    print(f"  Data: {args.data_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {device}")
    print(f"  Mouse weight: {args.mouse_weight}")
    print(f"  Turn action weight: {args.turn_action_weight}")
    print(f"{'='*60}\n")

    # 加载数据集
    dataset = EffectiveDataset(args.data_dir, max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # 检测num_goals
    all_goal_ids = set(s[5] for s in dataset.samples)
    num_goals = max(all_goal_ids) + 1 if all_goal_ids else 1
    print(f"[Train] Detected {num_goals} goals")

    # 初始化模型（输出10个动作，但只训练forward/right/left三个）
    model = GoalConditionedBC(num_goals=num_goals).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        avg_loss, acc, action_loss, mouse_loss = train_epoch(
            model, dataloader, optimizer, device, epoch,
            mouse_weight=args.mouse_weight,
            turn_action_weight=args.turn_action_weight,
            direction_weight=args.direction_weight
        )
        epoch_time = time.time() - start_time

        print(f"\n  Epoch {epoch}/{args.epochs} Loss: {avg_loss:.4f} Acc: {acc:.2f}% "
              f"Action: {action_loss:.4f} Mouse: {mouse_loss:.4f} Time: {epoch_time:.1f}s\n")

        # 保存checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"goal_bc_v4_epoch_{epoch:03d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [Save] {ckpt_path}")

    # 保存最终模型
    final_path = os.path.join(args.output_dir, "goal_bc_v4_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Model: {final_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--mouse-weight", type=float, default=10.0)
    parser.add_argument("--turn-action-weight", type=float, default=5.0)
    parser.add_argument("--direction-weight", type=float, default=0.5)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)