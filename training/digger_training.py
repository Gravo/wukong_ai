"""
digger_training.py - DAgger 训练脚本 v2.0 (懒加载)
加载人类数据 + DAgger纠正数据，重新训练模型
v2.0: 懒加载模式，避免OOM

Usage:
    C:\Python\python.exe -u training\digger_training.py ^
      --human-data pathfinding_data ^
      --dagger-data . ^
      --epochs 30 --batch-size 32 --lr 0.001
"""
import argparse, time, os, h5py, glob, json, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn.functional as F

from goal_conditioned_bc import GoalConditionedBC


class DAggerDataset(Dataset):
    """DAgger数据集：懒加载模式，避免OOM"""
    def __init__(self, human_data_dir, dagger_data_dir, max_samples=None, filter_idle=True):
        self.samples = []  # [(h5_path, frame_idx, action, mouse_dx, mouse_dy, goal_id, is_start)]
        self.mouse_dx_all = []
        self.mouse_dy_all = []
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        print(f"[DAggerDataset] Loading human data from {human_data_dir}...")
        for h5_file in sorted(glob.glob(os.path.join(human_data_dir, "*.h5"))):
            self._load_human_file(h5_file, filter_idle)
        print(f"[DAggerDataset] ✅ Human data: {len(self.samples)} samples")

        if dagger_data_dir and os.path.exists(dagger_data_dir):
            print(f"[DAggerDataset] Loading DAgger data from {dagger_data_dir}...")
            for h5_file in sorted(glob.glob(os.path.join(dagger_data_dir, "*.h5"))):
                self._load_dagger_file(h5_file)
            print(f"[DAggerDataset] ✅ DAgger data: {len(self.samples)} total samples")

        if max_samples and len(self.samples) > max_samples:
            print(f"[DAggerDataset] ⚠️  Truncating to {max_samples} samples")
            self.samples = self.samples[:max_samples]

        self.mouse_dx_all = np.array(self.mouse_dx_all, dtype=np.float32)
        self.mouse_dy_all = np.array(self.mouse_dy_all, dtype=np.float32)
        self.mouse_dx_std = self.mouse_dx_all.std() + 1e-8
        self.mouse_dy_std = self.mouse_dy_all.std() + 1e-8
        del self.mouse_dx_all, self.mouse_dy_all

        # Normalize mouse in-place
        normalized = []
        for h5, fidx, act, mdx, mdy, gid, ist in self.samples:
            normalized.append((h5, fidx, act, mdx/self.mouse_dx_std, mdy/self.mouse_dy_std, gid, ist))
        self.samples = normalized

        print(f"[DAggerDataset] ✅ Final: {len(self.samples)} samples")
        print(f"[DAggerDataset] 📊 Mouse std: dx={self.mouse_dx_std:.2f}, dy={self.mouse_dy_std:.2f}")

    def _load_human_file(self, h5_file, filter_idle):
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'frames' not in f: return
                actions = f['actions'][:]
                mouse_dx = f['mouse_dx'][:]
                mouse_dy = f['mouse_dy'][:]
                goal_id = f.attrs.get('goal_id', 0)
                indices = np.where(actions != 0)[0].tolist() if filter_idle else list(range(len(actions)))
                is_start_arr = np.zeros(len(actions), dtype=np.bool_)
                is_start_arr[:min(10, len(actions))] = True
                for i in indices:
                    self.samples.append((h5_file, int(i), int(actions[i]), float(mouse_dx[i]), float(mouse_dy[i]), int(goal_id), bool(is_start_arr[i])))
                    self.mouse_dx_all.append(float(mouse_dx[i]))
                    self.mouse_dy_all.append(float(mouse_dy[i]))
        except Exception as e:
            print(f"[DAggerDataset] ❌ {h5_file}: {e}")

    def _load_dagger_file(self, h5_file):
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'frames' not in f: return
                human_actions = f['human_actions'][:]
                mouse_dx = f['mouse_dx'][:]
                mouse_dy = f['mouse_dy'][:]
                goal_id = f.attrs.get('goal_id', 0)
                corrected = np.where(human_actions != -1)[0]
                print(f"  📊 {os.path.basename(h5_file)}: {len(corrected)} corrected samples")
                if len(corrected) == 0: return
                is_start_arr = np.zeros(len(human_actions), dtype=np.bool_)
                is_start_arr[:min(10, len(human_actions))] = True
                for i in corrected:
                    self.samples.append((h5_file, int(i), int(human_actions[i]), float(mouse_dx[i]), float(mouse_dy[i]), int(goal_id), bool(is_start_arr[i])))
                    self.mouse_dx_all.append(float(mouse_dx[i]))
                    self.mouse_dy_all.append(float(mouse_dy[i]))
        except Exception as e:
            print(f"[DAggerDataset] ❌ {h5_file}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_path, fidx, action, mouse_dx, mouse_dy, goal_id, is_start = self.samples[idx]
        with h5py.File(h5_path, 'r') as f:
            frame = f['frames'][fidx]
        if frame.shape[0] != 224 or frame.shape[1] != 224:
            frame = cv2.resize(frame, (224, 224))
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - self._mean) / self._std
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

        # 2. 鼠标回归损失
        mouse_criterion = nn.MSELoss(reduction='none')
        mouse_loss = mouse_criterion(mouse_pred, mouse_target)
        mouse_loss = mouse_loss.mean()

        # 3. 起始帧鼠标损失加权
        start_mask = is_start.float()
        start_ratio = start_mask.mean()
        start_mouse_loss = mouse_loss * start_ratio * start_mouse_weight

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
    print(f"  DAgger Training (v2.0 - Lazy Loading)")
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
    # 确保 goal_id 是 0-indexed
    all_goal_ids = set(s[5] for s in dataset.samples)
    min_gid = min(all_goal_ids) if all_goal_ids else 0
    if min_gid >= 1:  # 数据是 1-indexed，需要转换
        print(f"[Train] ⚠️  goal_ids are 1-indexed (min={min_gid}), converting to 0-indexed...", flush=True)
        # 转换 dataset.samples 中的 goal_id
        dataset.samples = [(h5, fidx, act, mdx, mdy, gid-1, ist) for (h5, fidx, act, mdx, mdy, gid, ist) in dataset.samples]
        all_goal_ids = set(s[5] for s in dataset.samples)
    
    num_goals = max(all_goal_ids) + 1 if all_goal_ids else 1
    print(f"[Train] Detected {num_goals} goals (0-indexed)", flush=True)
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
            checkpoint_path = os.path.join(args.output_dir, f"goal_bc_dagger_epoch_{epoch:03d}.pt")
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
    parser.add_argument("--human-data", type=str, required=True)
    parser.add_argument("--dagger-data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-filter-idle", action="store_true")
    parser.add_argument("--mouse-weight", type=float, default=10.0)
    parser.add_argument("--start-mouse-weight", type=float, default=20.0)
    parser.add_argument("--direction-weight", type=float, default=0.5)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
