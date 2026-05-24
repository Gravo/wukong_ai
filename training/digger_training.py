"""
dagger_training.py - DAgger 训练脚本 v1.0
加载人类数据 + DAgger纠正数据，重新训练模型

Usage:
    C:\Python\python.exe -u training\digger_training.py ^
      --human-data pathfinding:data ^
      --dagger-data dagger:data ^
      --epochs 30 --batch-size 32 --lr 0.001
"""
import argparse, time, os, h5py, glob, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn.functional as F

from goal_conditioned_bc import GoalConditionedBC


class DAggerDataset(Dataset):
    """DAgger数据集：支持人类数据 + DAgger纠正数据"""
    def __init__(self, human_data_dir, dagger_data_dir, max_samples=None, filter_idle=True):
        self.frames = []
        self.actions = []
        self.mouse_dx = []
        self.mouse_dy = []
        self.goal_ids = []
        self.is_start = []

        # 1. 加载人类数据
        print(f"[DAggerDataset] Loading human data from {human_data_dir}...")
        human_files = sorted(glob.glob(os.path.join(human_data_dir, "*.h5")))
        for h5_file in human_files:
            self._load_human_file(h5_file, filter_idle)
        print(f"[DAggerDataset] ✅ Human data: {len(self.frames)} samples")

        # 2. 加载DAgger纠正数据
        if dagger_data_dir and os.path.exists(dagger_data_dir):
            print(f"[DAggerDataset] Loading DAgger data from {dagger_data_dir}...")
            dagger_files = sorted(glob.glob(os.path.join(dagger_data_dir, "*.h5")))
            for h5_file in dagger_files:
                self._load_dagger_file(h5_file, filter_idle)
            print(f"[DAggerDataset] ✅ DAgger data: {len(self.frames)} total samples")

        # 3. 截断（可选）
        if max_samples and len(self.frames) > max_samples:
            print(f"[DAggerDataset] ⚠️  Truncating to {max_samples} samples")
            self.frames = self.frames[:max_samples]
            self.actions = self.actions[:max_samples]
            self.mouse_dx = self.mouse_dx[:max_samples]
            self.mouse_dy = self.mouse_dy[:max_samples]
            self.goal_ids = self.goal_ids[:max_samples]
            self.is_start = self.is_start[:max_samples]

        # 4. 计算鼠标标准差（用于归一化）
        self.mouse_dx = np.array(self.mouse_dx, dtype=np.float32)
        self.mouse_dy = np.array(self.mouse_dy, dtype=np.float32)
        self.mouse_dx_std = self.mouse_dx.std() + 1e-8
        self.mouse_dy_std = self.mouse_dy.std() + 1e-8
        self.mouse_dx = self.mouse_dx / self.mouse_dx_std
        self.mouse_dy = self.mouse_dy / self.mouse_dy_std

        print(f"[DAggerDataset] ✅ Final: {len(self.frames)} samples")
        print(f"[DAggerDataset] 📊 Mouse std: dx={self.mouse_dx_std:.2f}, dy={self.mouse_dy_std:.2f}")

    def _load_human_file(self, h5_file, filter_idle):
        """加载人类数据文件（标准格式）"""
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'frames' not in f:
                    return
                frames = f['frames'][:]
                actions = f['actions'][:]
                mouse_dx = f['mouse_dx'][:]
                mouse_dy = f['mouse_dy'][:]
                goal_id = f.attrs.get('goal_id', 0)

                # 过滤idle帧
                if filter_idle:
                    non_idle = actions != 0
                    frames = frames[non_idle]
                    actions = actions[non_idle]
                    mouse_dx = mouse_dx[non_idle]
                    mouse_dy = mouse_dy[non_idle]

                # 标记为起始帧（前10帧）
                is_start = np.zeros(len(frames), dtype=np.bool_)
                is_start[:min(10, len(frames))] = True

                self.frames.extend(frames)
                self.actions.extend(actions)
                self.mouse_dx.extend(mouse_dx)
                self.mouse_dy.extend(mouse_dy)
                self.goal_ids.extend([goal_id] * len(frames))
                self.is_start.extend(is_start)

        except Exception as e:
            print(f"[DAggerDataset] ❌ Error loading {h5_file}: {e}")

    def _load_dagger_file(self, h5_file, filter_idle):
        """加载DAgger数据文件（纠正数据）"""
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'frames' not in f:
                    return

                frames = f['frames'][:]
                ai_actions = f['ai_actions'][:]
                human_actions = f['human_actions'][:]
                mouse_dx = f['mouse_dx'][:]
                mouse_dy = f['mouse_dy'][:]
                intervention = f['intervention'][:]
                goal_id = f.attrs.get('goal_id', 0)

                # 只保留"干预帧"（人类纠正的数据）
                corrected = human_actions != -1
                frames = frames[corrected]
                actions = human_actions[corrected]  # 使用人类纠正的动作作为标签
                mouse_dx = mouse_dx[corrected]
                mouse_dy = mouse_dy[corrected]
                intervention = intervention[corrected]

                if len(frames) == 0:
                    return

                # 标记为起始帧
                is_start = np.zeros(len(frames), dtype=np.bool_)
                is_start[:min(10, len(frames))] = True

                self.frames.extend(frames)
                self.actions.extend(actions)
                self.mouse_dx.extend(mouse_dx)
                self.mouse_dy.extend(mouse_dy)
                self.goal_ids.extend([goal_id] * len(frames))
                self.is_start.extend(is_start)

                print(f"  📊 {os.path.basename(h5_file)}: {len(frames)} corrected samples")

        except Exception as e:
            print(f"[DAggerDataset] ❌ Error loading {h5_file}: {e}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        action = self.actions[idx]
        mouse_dx = self.mouse_dx[idx]
        mouse_dy = self.mouse_dy[idx]
        goal_id = self.goal_ids[idx]
        is_start = self.is_start[idx]

        # 预处理frame
        frame = frame.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        frame = (frame - mean) / std
        frame = frame.transpose(2, 0, 1)

        return (
            torch.from_numpy(frame),
            torch.tensor(action, dtype=torch.long),
            torch.tensor([mouse_dx, mouse_dy], dtype=torch.float32),
            torch.tensor(goal_id, dtype=torch.long),
            torch.tensor(is_start, dtype=torch.bool)
        )


def train_dagger(model, dataloader, optimizer, device, epoch, mouse_weight=10.0, start_mouse_weight=20.0, direction_weight=0.5):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_action_loss = 0
    total_mouse_loss = 0
    total_dir_loss = 0
    correct = 0
    total = 0

    for batch_idx, (frames, actions, mouse_target, goal_ids, is_start) in enumerate(dataloader):
        frames = frames.to(device)
        actions = actions.to(device)
        mouse_target = mouse_target.to(device)
        goal_ids = goal_ids.to(device)
        is_start = is_start.to(device)

        optimizer.zero_grad()

        action_logits, mouse_pred = model(frames, goal_ids)

        # 1. 动作分类损失
        action_loss = F.cross_entropy(action_logits, actions)

        # 2. 鼠标回归损失（支持逐样本加权）
        mouse_criterion = nn.MSELoss(reduction='none')
        mouse_loss = mouse_criterion(mouse_pred, mouse_target)
        mouse_loss = mouse_loss.mean()  # 简化：不加权

        # 3. 起始帧鼠标损失加权
        start_mouse_loss = mouse_loss * is_start.float().mean() * start_mouse_weight

        # 4. 方向一致性损失
        if mouse_pred.shape[0] > 1:
            direction_loss = -torch.mean(
                F.cosine_similarity(mouse_pred[:-1], mouse_pred[1:], dim=1)
            )
        else:
            direction_loss = torch.tensor(0.0, device=device)

        # 5. 总损失
        loss = action_loss + mouse_weight * mouse_loss + start_mouse_weight * start_mouse_loss + direction_weight * direction_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_action_loss += action_loss.item()
        total_mouse_loss += mouse_loss.item()
        total_dir_loss += direction_loss.item()

        _, predicted = torch.max(action_logits, 1)
        correct += (predicted == actions).sum().item()
        total += actions.size(0)

        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch} Batch {batch_idx}/{len(dataloader)} "
                  f"Loss: {loss.item():.4f} Mouse: {mouse_loss.item():.4f} Dir: {direction_loss.item():.4f}", flush=True)

    avg_loss = total_loss / len(dataloader)
    avg_action_loss = total_action_loss / len(dataloader)
    avg_mouse_loss = total_mouse_loss / len(dataloader)
    avg_dir_loss = total_dir_loss / len(dataloader)
    acc = 100.0 * correct / total

    return avg_loss, acc, avg_action_loss, avg_mouse_loss, avg_dir_loss


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  DAgger Training (v1.0)")
    print(f"  Human data: {args.human_data}")
    print(f"  DAgger data: {args.dagger_data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # 1. 加载数据集
    dataset = DAggerDataset(
        human_data_dir=args.human_data,
        dagger_data_dir=args.dagger_data,
        max_samples=args.max_samples,
        filter_idle=not args.no_filter_idle
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # 2. 初始化模型
    num_goals = len(set(dataset.goal_ids)) if dataset.goal_ids else 3
    model = GoalConditionedBC(num_goals=num_goals).to(device)

    # 3. 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. 训练循环
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        avg_loss, acc, action_loss, mouse_loss, dir_loss = train_dagger(
            model, dataloader, optimizer, device, epoch,
            mouse_weight=args.mouse_weight,
            start_mouse_weight=args.start_mouse_weight,
            direction_weight=args.direction_weight
        )
        epoch_time = time.time() - start_time

        print(f"\n  Epoch {epoch}/{args.epochs} "
              f"Loss: {avg_loss:.4f} Acc: {acc:.2f}% "
              f"Mouse: {mouse_loss:.4f} Dir: {dir_loss:.4f} "
              f"Time: {epoch_time:.1f}s\n")

        # 保存checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f"goal_bc_epoch_{epoch:03d}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [Save] Checkpoint saved: {checkpoint_path}", flush=True)

    # 5. 保存最终模型
    final_path = os.path.join(args.output_dir, "goal_bc_dagger_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Final model saved to: {final_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-data", type=str, required=True, help="Directory with human data (h5 files)")
    parser.add_argument("--dagger-data", type=str, default=None, help="Directory with DAgger data (h5 files)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples (truncate)")
    parser.add_argument("--no-filter-idle", action="store_true", help="Disable idle frame filtering")
    parser.add_argument("--mouse-weight", type=float, default=10.0, help="Mouse loss weight")
    parser.add_argument("--start-mouse-weight", type=float, default=20.0, help="Start frame mouse loss weight")
    parser.add_argument("--direction-weight", type=float, default=0.5, help="Direction consistency loss weight")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
