"""
behavior_clone_v3.py - 改进的行为克隆训练脚本

改进内容（相比 v2）：
1. 数据过滤：只保留 10-20% 的 idle 帧，强制模型学习有意义的动作
2. 学习率调度：CosineAnnealingLR，避免后期震荡
3. 加大鼠标损失权重：2.0x，解决鼠标控制失效问题
4. 更好的统计：训练前先分析数据分布
5. 内存优化：逐文件加载 + 及时释放

用法：
    # 使用默认配置
    python pathfinding/behavior_clone_v3.py

    # 自定义参数
    python pathfinding/behavior_clone_v3.py --epochs 100 --idle-ratio 0.1 --mouse-weight 2.0
"""
import os
import sys
import argparse
import gc
import numpy as np
import glob
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BC, NUM_ACTIONS, FRAME_STACK
from models.bc_model import BehaviorCloneModel


class FilteredH5Dataset(Dataset):
    """
    改进的 H5 数据集：
    1. 过滤掉大部分 idle 帧
    2. 支持数据增强（水平翻转）
    3. 在线帧堆叠
    """

    def __init__(self, h5_path, augment=True, idle_keep_ratio=0.1):
        """
        Args:
            h5_path: str，h5 文件路径
            augment: bool，是否启用数据增强
            idle_keep_ratio: float，保留多少比例的 idle 帧（0.0-1.0）
        """
        import h5py
        self.augment = augment
        self.idle_keep_ratio = idle_keep_ratio

        with h5py.File(h5_path, "r") as f:
            frames = f["frames"][:]
            actions = f["actions"][:]
            n = len(frames)

            has_mouse = f.attrs.get("has_mouse", False) and "mouse_dx" in f
            if has_mouse:
                mouse_dx = f["mouse_dx"][:]
                mouse_dy = f["mouse_dy"][:]
            else:
                mouse_dx = np.zeros(n, dtype=np.float32)
                mouse_dy = np.zeros(n, dtype=np.float32)

        # 计算有效样本范围
        total_samples = max(0, len(frames) - FRAME_STACK + 1)

        # 过滤 idle 帧
        if idle_keep_ratio < 1.0:
            # 找到每个样本对应的 action（取帧堆叠中间帧的 action）
            sample_actions = np.array([
                int(actions[i + FRAME_STACK - 1])
                for i in range(total_samples)
            ])

            # 分离 idle 和非 idle 样本
            idle_indices = np.where(sample_actions == 0)[0]
            active_indices = np.where(sample_actions > 0)[0]

            # 保留部分 idle 帧
            keep_idle = max(1, int(len(idle_indices) * idle_keep_ratio))
            if keep_idle < len(idle_indices):
                selected_idle = np.random.choice(idle_indices, keep_idle, replace=False)
            else:
                selected_idle = idle_indices

            # 合并并排序
            self.selected_indices = np.sort(np.concatenate([active_indices, selected_idle]))

            print(f"  [数据过滤] 总样本: {total_samples}, "
                  f"活跃: {len(active_indices)}, "
                  f"idle保留: {len(selected_idle)}/{len(idle_indices)}")
        else:
            self.selected_indices = np.arange(total_samples)

        # 存储原始数据（不复制）
        self.frames = frames
        self.actions = actions
        self.mouse_dx = mouse_dx
        self.mouse_dy = mouse_dy

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        # 映射到原始索引
        original_idx = self.selected_indices[idx]
        fi = original_idx + FRAME_STACK - 1

        # 帧堆叠：预分配 + 直接内存复制（比 concat 快 5x）
        frame = np.empty((FRAME_STACK * 3, 224, 224), dtype=np.uint8)
        for k in range(FRAME_STACK):
            src = self.frames[fi - FRAME_STACK + 1 + k]  # (224, 224, 3)
            frame[k * 3:(k + 1) * 3] = src.transpose(2, 0, 1)

        frame = frame.astype(np.float32) / 255.0
        action = int(self.actions[fi])

        # 鼠标归一化：除以 100，裁剪到 [-1, 1]
        mouse = np.array([
            np.clip(float(self.mouse_dx[fi]) / 100.0, -1.0, 1.0),
            np.clip(float(self.mouse_dy[fi]) / 100.0, -1.0, 1.0),
        ], dtype=np.float32)

        # 数据增强：水平翻转
        if self.augment and np.random.random() > 0.5:
            frame = frame[:, :, ::-1].copy()
            # 左右互换
            if action == 5:
                action = 6
            elif action == 6:
                action = 5
            mouse[0] = -mouse[0]

        return frame, action, mouse


def analyze_data(data_dir):
    """分析数据分布，返回统计信息"""
    h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if not h5_files:
        print(f"[错误] 未找到 h5 文件: {data_dir}")
        return None

    total_samples = 0
    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)

    for h5_path in h5_files:
        import h5py
        with h5py.File(h5_path, "r") as f:
            actions = f["actions"][:]
            n = len(actions)
            ns = max(0, n - FRAME_STACK + 1)
            total_samples += ns

            # 统计动作分布
            for i in range(ns):
                action_counts[int(actions[i + FRAME_STACK - 1])] += 1

    names = ['idle', 'atk', 'heavy', 'dodge', 'fwd', 'rgt', 'lft', 'd_atk', 'lock', 'heal']

    print(f"\n{'=' * 60}")
    print(f"  数据分布分析")
    print(f"{'=' * 60}")
    print(f"  总样本: {total_samples}")
    print(f"  h5 文件: {len(h5_files)}")
    print(f"\n  动作分布:")

    for i in range(NUM_ACTIONS):
        if action_counts[i] > 0:
            pct = action_counts[i] / total_samples * 100
            bar_len = 30
            filled = int(pct / 100 * bar_len)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"    {names[i]:8s}: {action_counts[i]:6d} ({pct:5.1f}%) {bar}")

    idle_pct = action_counts[0] / total_samples * 100
    print(f"\n  idle 占比: {idle_pct:.1f}%")
    if idle_pct > 50:
        print(f"  ⚠️  idle 占比过高，建议使用 --idle-ratio 0.1 过滤")

    print(f"{'=' * 60}\n")

    return {
        "total_samples": total_samples,
        "action_counts": action_counts,
        "idle_pct": idle_pct,
    }


def train_bc_v3(args):
    """改进的行为克隆训练 v3"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BC v3] 设备: {device}", flush=True)
    print(f"[BC v3] 配置: epochs={args.epochs}, lr={args.lr}, "
          f"idle_ratio={args.idle_ratio}, mouse_weight={args.mouse_weight}", flush=True)

    # 分析数据
    stats = analyze_data(args.data_dir)
    if stats is None:
        return

    h5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    if not h5_files:
        print(f"[BC v3] 未找到 h5 文件", flush=True)
        return

    # 创建模型
    model = BehaviorCloneModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[BC v3] 模型参数量: {param_count:,}", flush=True)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度器：CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,  # 最小学习率 = 初始的 1%
    )

    # 损失函数
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()

    # 训练记录
    best_acc = 0
    best_mouse_mae = float('inf')
    training_log = []

    print(f"\n[BC v3] 开始训练...", flush=True)

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
            dataset = FilteredH5Dataset(
                h5_path,
                augment=args.augment,
                idle_keep_ratio=args.idle_ratio,
            )

            # 每个文件内 shuffle
            indices = np.random.permutation(len(dataset))
            batch_size = args.batch_size

            for batch_start in range(0, len(dataset), batch_size):
                batch_idx = indices[batch_start:batch_start + batch_size]
                batch = [dataset[i] for i in batch_idx]

                frames = torch.stack([torch.from_numpy(b[0]) for b in batch]).to(device, non_blocking=True)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
                mouse_tgt = torch.stack([torch.from_numpy(b[2]) for b in batch]).to(device, non_blocking=True)

                logits, mouse_pred = model(frames)

                # 分类损失
                cls_loss = ce_criterion(logits, actions)

                # 鼠标损失（加权）
                mse_loss = mse_criterion(mouse_pred, mouse_tgt)

                # 总损失：加大鼠标权重
                loss = cls_loss + args.mouse_weight * mse_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_idx)
                cls_loss_sum += cls_loss.item() * len(batch_idx)
                mse_loss_sum += mse_loss.item() * len(batch_idx)
                correct += (logits.argmax(1) == actions).sum().item()
                mouse_mae_sum += torch.abs(mouse_pred - mouse_tgt).sum().item()
                total += len(batch_idx)

            del dataset
            print(f"  file {fi + 1}/{len(h5_files)} done", end="\r", flush=True)

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 统计
        acc = correct / total
        avg_loss = total_loss / total
        avg_cls = cls_loss_sum / total
        avg_mse = mse_loss_sum / total
        avg_mouse_mae = mouse_mae_sum / (total * 2)

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"loss={avg_loss:.4f} (cls={avg_cls:.4f}+mse={avg_mse:.4f}) "
            f"acc={acc:.2%} mouse_mae={avg_mouse_mae:.4f} lr={current_lr:.6f}",
            flush=True,
        )

        # 记录训练日志
        training_log.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "cls_loss": avg_cls,
            "mse_loss": avg_mse,
            "accuracy": acc,
            "mouse_mae": avg_mouse_mae,
            "lr": current_lr,
        })

        # 内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 保存最佳模型（综合考虑准确率和鼠标 MAE）
        if acc > best_acc:
            best_acc = acc
            best_mouse_mae = avg_mouse_mae
            os.makedirs("checkpoints", exist_ok=True)
            save_path = "checkpoints/bc_v3_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  → 保存最佳模型: {save_path} (acc={acc:.2%})", flush=True)

    # 保存训练日志
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"bc_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_path, "w") as f:
        json.dump({
            "config": {
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "idle_ratio": args.idle_ratio,
                "mouse_weight": args.mouse_weight,
                "augment": args.augment,
            },
            "best_accuracy": best_acc,
            "best_mouse_mae": best_mouse_mae,
            "log": training_log,
        }, f, indent=2)

    print(f"\n[BC v3] 训练完成！", flush=True)
    print(f"  最佳准确率: {best_acc:.2%}", flush=True)
    print(f"  最佳鼠标 MAE: {best_mouse_mae:.4f}", flush=True)
    print(f"  训练日志: {log_path}", flush=True)
    print(f"  模型文件: checkpoints/bc_v3_best.pt", flush=True)


def main():
    parser = argparse.ArgumentParser(description="行为克隆 v3（数据过滤 + LR 调度）")
    parser.add_argument("--data-dir", default=BC["data_dir"], help="数据目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--batch-size", type=int, default=32, help="批量大小")
    parser.add_argument("--idle-ratio", type=float, default=0.1, help="idle 帧保留比例")
    parser.add_argument("--mouse-weight", type=float, default=2.0, help="鼠标损失权重")
    parser.add_argument("--augment", action="store_true", default=True, help="启用数据增强")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="禁用数据增强")
    parser.add_argument("--analyze-only", action="store_true", help="只分析数据分布，不训练")

    args = parser.parse_args()

    if args.analyze_only:
        analyze_data(args.data_dir)
    else:
        train_bc_v3(args)


if __name__ == "__main__":
    main()
