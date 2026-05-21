"""
自动闪避模块训练脚本

训练一个轻量级 CNN 模型，检测敌人攻击前摇

数据格式：
- 输入：4帧堆叠画面 (12, 224, 224)
- 输出：是否应该闪避 (0/1)

使用方式：
    python assist/train_dodge.py --data l2_data/dodge_data_*.h5
"""
import os
import sys
import glob
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assist.auto_dodge import AttackDetector


class DodgeDataset(Dataset):
    """闪避数据集"""

    def __init__(self, h5_path: str, augment: bool = True):
        import h5py

        self.augment = augment

        with h5py.File(h5_path, "r") as f:
            self.frames = f["frames"][:]  # (N, 12, 224, 224)
            self.labels = f["labels"][:]  # (N,)

        # 归一化
        self.frames = self.frames.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]

        # 数据增强：随机水平翻转
        if self.augment and np.random.random() > 0.5:
            frame = frame[:, :, ::-1].copy()

        return torch.from_numpy(frame), torch.tensor(label, dtype=torch.float32)


def train(args):
    """训练闪避检测模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] 设备: {device}")

    # 查找数据文件
    h5_files = sorted(glob.glob(args.data))
    if not h5_files:
        print(f"[错误] 未找到数据文件: {args.data}")
        return

    print(f"[Train] 找到 {len(h5_files)} 个数据文件")

    # 加载数据
    datasets = [DodgeDataset(f, augment=True) for f in h5_files]
    dataset = torch.utils.data.ConcatDataset(datasets)

    print(f"[Train] 总样本数: {len(dataset)}")

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 创建模型
    model = AttackDetector().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Train] 模型参数量: {param_count:,}")

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # 训练
    best_val_acc = 0

    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for frames, labels in train_loader:
            frames = frames.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(frames).squeeze()
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * len(labels)
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += len(labels)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)

                outputs = model(frames).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item() * len(labels)
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += len(labels)

        # 更新学习率
        scheduler.step()

        # 打印统计
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_loss /= train_total
        val_loss /= val_total

        print(f"Epoch {epoch + 1}/{args.epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            save_path = "checkpoints/auto_dodge_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  → 保存最佳模型: {save_path} (val_acc={val_acc:.2%})")

    print(f"\n[Train] 训练完成!")
    print(f"  最佳验证准确率: {best_val_acc:.2%}")
    print(f"  模型文件: checkpoints/auto_dodge_best.pt")


def main():
    parser = argparse.ArgumentParser(description="训练自动闪避模型")
    parser.add_argument("--data", default="l2_data/dodge_data_*.h5",
                        help="数据文件路径（支持通配符）")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
