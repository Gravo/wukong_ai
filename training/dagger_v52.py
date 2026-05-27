"""
dagger_v52.py - DAgger 训练脚本（v5.2 集成版）

DAgger (Dataset Aggregation) 解决协变量漂移问题：
1. 用当前模型运行推理
2. 人类纠正模型的错误决策
3. 将纠正数据加入训练集
4. 重新训练模型
5. 重复直到干预率 < 25%

用法：
    # 第 1 轮：用 v5.2 模型采集纠正数据
    python training/dagger_v52.py --model checkpoints/goal_bc_v52_best.pt --round 1

    # 第 2 轮：用第 1 轮训练的模型继续
    python training/dagger_v52.py --model checkpoints/dagger_round1_best.pt --round 2
"""
import os
import sys
import argparse
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_FRAMES = 4
STACK_OFFSETS = [0, 1, 3, 7]
NUM_CLASSES = 5
ACTION_NAMES = ["idle", "forward", "turn_slow", "turn_medium", "turn_fast"]


class V52Model(nn.Module):
    """v5.2 模型架构"""
    def __init__(self, num_goals, num_frames=NUM_FRAMES, pretrained=False):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3 * num_frames, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.goal_embed = nn.Embedding(num_goals, 64)
        self.fc = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x, goal_id):
        B = x.shape[0]
        vis = self.backbone(x).view(B, -1)
        g = self.goal_embed(goal_id)
        return self.fc(torch.cat([vis, g], dim=1))


class DAggerDataset(Dataset):
    """DAgger 数据集（懒加载）"""
    def __init__(self, data_dir, dagger_file=None):
        self.samples = []
        self.file_meta = {}
        self._build(data_dir, dagger_file)

    def _build(self, data_dir, dagger_file):
        # 加载原始数据
        data_path = Path(data_dir)
        for h5_file in data_path.glob("*.h5"):
            self._add_file(h5_file)

        # 加载 DAgger 数据
        if dagger_file and os.path.exists(dagger_file):
            self._add_file(Path(dagger_file))

        print(f"[DAgger Dataset] {len(self.samples)} samples")

    def _add_file(self, h5_file):
        with h5py.File(h5_file, 'r') as f:
            n = len(f['actions'])
            max_offset = max(STACK_OFFSETS)
            for i in range(max_offset, n):
                self.samples.append((str(h5_file), i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_path, frame_idx = self.samples[idx]

        if h5_path not in self.file_meta:
            self.file_meta[h5_path] = h5py.File(h5_path, 'r')

        f = self.file_meta[h5_path]
        frames = f['frames']
        actions = f['actions']
        goals = f['goals'] if 'goals' in f else None

        # 非均匀堆叠
        stacked = []
        for offset in STACK_OFFSETS:
            idx = frame_idx - offset
            if idx < 0:
                stacked.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                stacked.append(frames[idx])

        stacked = np.concatenate([cv2.resize(f, (224, 224)).transpose(2, 0, 1) for f in stacked], axis=0)
        stacked = stacked.astype(np.float32) / 255.0

        action = int(actions[frame_idx])
        goal = int(goals[frame_idx]) if goals is not None else 0

        return torch.from_numpy(stacked), torch.tensor(action, dtype=torch.long), torch.tensor(goal, dtype=torch.long)


def train_epoch(model, dl, opt, device, epoch, fl_fn):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (x, y, g) in enumerate(dl):
        x, y, g = x.to(device), y.to(device), g.to(device)

        opt.zero_grad()
        pred = model(x, g)
        loss = fl_fn(pred, y)
        loss.backward()
        opt.step()

        total_loss += loss.item() * len(x)
        correct += (pred.argmax(1) == y).sum().item()
        total += len(x)

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: Loss={loss.item():.4f} Acc={100*correct/total:.2f}%")

    return total_loss / total, 100 * correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  DAgger Training (v5.2)")
    print(f"  Round: {args.round}")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # 加载模型
    model = V52Model(num_goals=args.num_goals).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"[DAgger] 模型加载成功: {args.model}")

    # 创建数据集
    dagger_file = f"pathfinding_dagger_round{args.round}.h5"
    ds = DAggerDataset(args.data_dir, dagger_file)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 损失函数
    fl_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # 训练
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, dl, opt, device, epoch, fl_fn)

        if acc > best_acc:
            best_acc = acc
            # 保存最佳模型
            best_ckpt = os.path.join(args.output_dir, f"dagger_round{args.round}_best.pt")
            torch.save(model.state_dict(), best_ckpt)
            print(f"  [Save Best] {best_ckpt} (Acc={acc:.2f}%)")

        print(f"\n  Epoch {epoch}/{args.epochs}  Loss:{loss:.4f}  Acc:{acc:.2f}%  Best:{best_acc:.2f}%\n")

    print(f"\n[DAgger] 训练完成！最佳准确率: {best_acc:.2f}%")
    print(f"[DAgger] 下一步: 用新模型运行推理，采集第 {args.round + 1} 轮纠正数据")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DAgger 训练脚本 (v5.2)")
    p.add_argument("--model", required=True, help="模型路径")
    p.add_argument("--data-dir", default="pathfinding_data", help="数据目录")
    p.add_argument("--round", type=int, default=1, help="DAgger 轮次")
    p.add_argument("--epochs", type=int, default=20, help="训练轮数")
    p.add_argument("--batch-size", type=int, default=4, help="批量大小")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--num-goals", type=int, default=2, help="目标数量")
    p.add_argument("--output-dir", default="checkpoints", help="输出目录")
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    main(a)
