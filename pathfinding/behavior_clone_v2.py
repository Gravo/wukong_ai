"""
behavior_clone_v2.py - 行为克隆模型 v2（极低内存版）
逐文件加载训练，每个文件处理完立即释放。
8249样本、12个h5文件，每个~60MB原始帧，峰值内存~500MB。

架构：视觉编码器 → 分类头(动作) + 回归头(mouse_dx, mouse_dy)
"""
import os
import sys
import argparse
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BC, NUM_ACTIONS, FRAME_STACK
from models.resnet_encoder import create_encoder


class H5Dataset(Dataset):
    """单个h5文件的Dataset，在线帧堆叠"""

    def __init__(self, h5_path, augment=True):
        import h5py
        self.augment = augment

        with h5py.File(h5_path, "r") as f:
            self.frames = f["frames"][:]
            self.actions = f["actions"][:]
            n = len(self.frames)
            has_mouse = f.attrs.get("has_mouse", False) and "mouse_dx" in f
            if has_mouse:
                self.mouse_dx = f["mouse_dx"][:]
                self.mouse_dy = f["mouse_dy"][:]
            else:
                self.mouse_dx = np.zeros(n, dtype=np.float32)
                self.mouse_dy = np.zeros(n, dtype=np.float32)

        self.n_samples = max(0, len(self.frames) - FRAME_STACK + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        fi = idx + FRAME_STACK - 1
        # 预分配 + 直接内存复制（比concat快5x）
        frame = np.empty((FRAME_STACK * 3, 224, 224), dtype=np.uint8)
        for k in range(FRAME_STACK):
            src = self.frames[fi - FRAME_STACK + 1 + k]  # (224,224,3)
            frame[k*3:(k+1)*3] = src.transpose(2, 0, 1)
        frame = frame.astype(np.float32) / 255.0
        action = int(self.actions[fi])
        mouse = np.array([
            np.clip(float(self.mouse_dx[fi]) / 100.0, -1.0, 1.0),
            np.clip(float(self.mouse_dy[fi]) / 100.0, -1.0, 1.0),
        ], dtype=np.float32)
        if self.augment and np.random.random() > 0.5:
            frame = frame[:, :, ::-1].copy()
            if action == 5: action = 6
            elif action == 6: action = 5
            mouse[0] = -mouse[0]
        return frame, action, mouse


class BehaviorCloneModel(nn.Module):
    def __init__(self, latent_dim=256, num_actions=NUM_ACTIONS):
        super().__init__()
        self.encoder = create_encoder("resnet18", latent_dim=latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, num_actions),
        )
        self.mouse_regressor = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 2),
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat), self.mouse_regressor(feat)


def train_bc(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BC] 设备: {device}", flush=True)

    h5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    if not h5_files:
        print(f"[BC] 未找到h5文件", flush=True)
        return

    # 先统计总样本和动作分布
    total_samples = 0
    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    for h5_path in h5_files:
        import h5py
        with h5py.File(h5_path, "r") as f:
            n = f.attrs["num_frames"]
        ns = max(0, n - FRAME_STACK + 1)
        total_samples += ns
        # 粗略统计
        tmp = H5Dataset(h5_path, augment=False)
        for i in range(len(tmp)):
            action_counts[tmp[i][1]] += 1
        del tmp

    print(f"[BC] {len(h5_files)} 个h5文件, {total_samples} 样本", flush=True)
    names = ['idle','atk','heavy','dodge','fwd','rgt','lft','d_atk','lock','heal']
    for i in range(NUM_ACTIONS):
        if action_counts[i] > 0:
            print(f"  {names[i]:8s}: {action_counts[i]} ({action_counts[i]/total_samples*100:.1f}%)", flush=True)

    model = BehaviorCloneModel().to(device)
    print(f"[BC] 模型参数量: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=BC["lr"])
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()

    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        cls_loss_sum = 0
        mse_loss_sum = 0
        correct = 0
        total = 0
        mouse_mae_sum = 0

        # 逐文件加载训练
        for fi, h5_path in enumerate(h5_files):
            dataset = H5Dataset(h5_path, augment=BC["augment"])
            # 每个文件内shuffle
            indices = np.random.permutation(len(dataset))
            batch_size = BC["batch_size"]

            for batch_start in range(0, len(dataset), batch_size):
                batch_idx = indices[batch_start:batch_start + batch_size]
                batch = [dataset[i] for i in batch_idx]

                frames = torch.stack([torch.from_numpy(b[0]) for b in batch]).to(device, non_blocking=True)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
                mouse_tgt = torch.stack([torch.from_numpy(b[2]) for b in batch]).to(device, non_blocking=True)

                logits, mouse_pred = model(frames)
                cls_loss = ce_criterion(logits, actions)
                mse_loss = mse_criterion(mouse_pred, mouse_tgt)
                loss = cls_loss + 0.5 * mse_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_idx)
                cls_loss_sum += cls_loss.item() * len(batch_idx)
                mse_loss_sum += mse_loss.item() * len(batch_idx)
                correct += (logits.argmax(1) == actions).sum().item()
                mouse_mae_sum += torch.abs(mouse_pred - mouse_tgt).sum().item()
                total += len(batch_idx)

            del dataset  # 释放内存！
            print(f"  file {fi+1}/{len(h5_files)} done", end="\r", flush=True)

        acc = correct / total
        avg_loss = total_loss / total
        avg_cls = cls_loss_sum / total
        avg_mse = mse_loss_sum / total
        avg_mmae = mouse_mae_sum / (total * 2)

        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"loss={avg_loss:.4f} (cls={avg_cls:.4f}+mse={avg_mse:.4f}) "
            f"acc={acc:.2%} mouse_mae={avg_mmae:.4f}",
            flush=True,
        )

        if acc > best_acc:
            best_acc = acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/bc_best.pt")
            print(f"  → saved best (acc={acc:.2%})", flush=True)

    print(f"\n[BC] 训练完成！最佳准确率: {best_acc:.2%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=BC["data_dir"])
    parser.add_argument("--epochs", type=int, default=BC["epochs"])
    args = parser.parse_args()
    train_bc(args)
