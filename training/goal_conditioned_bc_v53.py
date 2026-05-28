"""
goal_conditioned_bc_v53.py - v5.3 训练脚本

改进（相比 v5.2）：
  1. 双头输出：Action Head (3-class) + Mouse Head (2-regression)
  2. 连续转角输出：mouse_dx/mouse_dy 回归
  3. 过滤 idle 帧（从 filtered 数据加载）
  4. 非均匀帧堆叠 [0,1,3,7]
  5. 懒加载（避免 OOM）

架构：
  Input: 4帧堆叠 (12通道) + goal_id (0/1)
    ↓
  ResNet18 (pretrained) + AdaptiveAvgPool2d
    ↓
  Feature Vector (512-dim)
    ↓
  ┌─────────────────┬─────────────────┐
  │ Action Head     │ Mouse Head      │
  │ (3-class)      │ (2 regression) │
  │ - forward       │ - mouse_dx      │
  │ - turn_left     │ - mouse_dy      │
  │ - turn_right    │                 │
  └─────────────────┴─────────────────┘

Loss：
  action_loss = CrossEntropyLoss(weights=[1.0, 5.0, 5.0])
  mouse_loss = MSELoss()
  total_loss = action_loss + 10.0 * mouse_loss

使用方法：
  C:\Python\python.exe -u training\goal_conditioned_bc_v53.py ^
    --data-dir "D:\projects\wukong_ai\pathfinding_data_noidle" ^
    --output-dir "D:\projects\wukong_ai\checkpoints" ^
    --batch-size 4 ^
    --epochs 50 ^
    --lr 1e-4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import torchvision.models as models

# ============ 配置 ============

ACTION_FORWARD = 4
ACTION_TURN_LEFT = 5
ACTION_TURN_RIGHT = 6

# 动作映射（3类）
ACTION_MAP = {
    ACTION_FORWARD: 0,      # forward -> class 0
    ACTION_TURN_LEFT: 1,     # turn_left -> class 1
    ACTION_TURN_RIGHT: 2,     # turn_right -> class 2
}

# 帧堆叠偏移
FRAME_OFFSETS = [0, 1, 3, 7]  # 非均匀堆叠

# ====================================


class FilteredH5Dataset(Dataset):
    """
    懒加载 Dataset - 从过滤后数据加载（无 idle）
    双头标签：action class + mouse_dx/mouse_dy
    """
    
    def __init__(self, data_dir, frame_offsets=FRAME_OFFSETS, max_samples=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.frame_offsets = frame_offsets
        self.max_samples = max_samples
        
        # 扫描所有 h5 文件
        self.h5_files = sorted(self.data_dir.glob("*.h5"))
        if len(self.h5_files) == 0:
            raise RuntimeError(f"在 {data_dir} 中没有找到 .h5 文件")
        
        # 构建索引： (file_idx, frame_idx)
        self.index_map = []
        self.file_lengths = []
        
        print(f"扫描数据目录: {data_dir}", flush=True)
        print(f"找到 {len(self.h5_files)} 个 .h5 文件", flush=True)
        
        for file_idx, h5_path in enumerate(self.h5_files):
            with h5py.File(h5_path, "r") as f:
                n_frames = len(f["frames"])
                self.file_lengths.append(n_frames)
                
                # 有效索引：frame_idx 必须 >= max(frame_offsets)
                min_idx = max(frame_offsets)
                for frame_idx in range(min_idx, n_frames):
                    self.index_map.append((file_idx, frame_idx))
        
        if self.max_samples is not None:
            self.index_map = self.index_map[:self.max_samples]
        
        print(f"总样本数: {len(self.index_map)}", flush=True)
        print(f"帧堆叠偏移: {frame_offsets}", flush=True)
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        file_idx, frame_idx = self.index_map[idx]
        h5_path = self.h5_files[file_idx]
        
        with h5py.File(h5_path, "r") as f:
            # 加载多帧 (C, H, W)
            frames = []
            for offset in self.frame_offsets:
                actual_idx = frame_idx - offset
                frame = f["frames"][actual_idx]  # (H, W, C)
                frame = frame.transpose(2, 0, 1)  # (C, H, W)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            
            # 堆叠 (N*C, H, W)
            stacked_frames = np.concatenate(frames, axis=0)  # (12, H, W)
            
            # 动作标签（3类）
            action = f["actions"][frame_idx]
            action_class = ACTION_MAP.get(action, 0)  # 默认 forward
            
            # 鼠标标签（回归）
            mouse_dx = f["mouse_dx"][frame_idx] if "mouse_dx" in f else 0.0
            mouse_dy = f["mouse_dy"][frame_idx] if "mouse_dy" in f else 0.0
            
            # Goal ID
            goal_id = f["goal_ids"][frame_idx] if "goal_ids" in f else 0
        
        return {
            "frames": stacked_frames,  # (12, H, W)
            "action_class": action_class,  # 0/1/2
            "mouse_dx": mouse_dx,  # 连续值
            "mouse_dy": mouse_dy,  # 连续值
            "goal_id": goal_id,  # 0/1
        }


class DualHeadModel(nn.Module):
    """
    双头模型：Action Head (分类) + Mouse Head (回归)
    """
    
    def __init__(self, num_actions=3, latent_dim=512, hidden_dim=256):
        super().__init__()
        
        # ResNet18 骨干（修改第一层以支持 12 通道）
        self.backbone = models.resnet18(pretrained=True)
        
        # 修改第一层卷积（从 3 通道 -> 12 通道）
        old_conv1 = self.backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=12,  # 4帧 * 3通道
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False
        )
        
        # 初始化新卷积层
        with torch.no_grad():
            # 前 3 通道用预训练权重
            new_conv1.weight[:, :3, :, :] = old_conv1.weight
            # 后 9 通道用前 3 通道的拷贝（加小扰动）
            for i in range(1, 4):
                new_conv1.weight[:, i*3:(i+1)*3, :, :] = \
                    old_conv1.weight + torch.randn_like(old_conv1.weight) * 0.01
        
        self.backbone.conv1 = new_conv1
        
        # 移除最后的分类层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Goal embedding
        self.goal_embed = nn.Embedding(2, 32)  # 2个goal -> 32-dim
        
        # 特征维度
        self.feature_dim = 512 + 32  # ResNet18 output + goal embedding
        
        # Action Head (3-class)
        self.action_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_actions),
        )
        
        # Mouse Head (2-regression)
        self.mouse_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),  # mouse_dx, mouse_dy
        )
    
    def forward(self, frames, goal_ids):
        """
        Args:
            frames: (B, 12, H, W)
            goal_ids: (B,) - 需要是 LongTensor
        Returns:
            action_logits: (B, 3)
            mouse_pred: (B, 2)
        """
        # 确保 goal_ids 是 LongTensor
        goal_ids = goal_ids.long()
        # 骨干网络
        B = frames.shape[0]
        x = self.backbone(frames).view(B, -1)  # (B, 512)
        
        # Goal embedding
        goal_emb = self.goal_embed(goal_ids)  # (B, 32)
        
        # 拼接
        features = torch.cat([x, goal_emb], dim=-1)  # (B, 544)
        
        # 双头输出
        action_logits = self.action_head(features)  # (B, 3)
        mouse_pred = self.mouse_head(features)  # (B, 2)
        
        return action_logits, mouse_pred


def train_epoch(model, dataloader, optimizer, device, class_weights=None):
    """
    训练一个 epoch
    """
    model.train()
    total_loss = 0.0
    total_action_loss = 0.0
    total_mouse_loss = 0.0
    correct = 0
    total = 0
    
    # 动作损失权重
    if class_weights is not None:
        action_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        action_criterion = nn.CrossEntropyLoss()
    
    # 鼠标损失（MSE）
    mouse_criterion = nn.MSELoss()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        frames = batch["frames"].to(device)  # (B, 12, H, W)
        goal_ids = batch["goal_id"].to(device)
        action_classes = batch["action_class"].to(device)
        mouse_dx = batch["mouse_dx"].to(device).unsqueeze(-1)
        mouse_dy = batch["mouse_dy"].to(device).unsqueeze(-1)
        mouse_targets = torch.cat([mouse_dx, mouse_dy], dim=-1)  # (B, 2)
        
        # 前向传播
        action_logits, mouse_pred = model(frames, goal_ids)
        
        # 计算损失
        action_loss = action_criterion(action_logits, action_classes)
        mouse_loss = mouse_criterion(mouse_pred, mouse_targets)
        
        # 总损失（加权）
        loss = action_loss + 10.0 * mouse_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 统计
        total_loss += loss.item() * frames.shape[0]
        total_action_loss += action_loss.item() * frames.shape[0]
        total_mouse_loss += mouse_loss.item() * frames.shape[0]
        
        _, predicted = action_logits.max(1)
        correct += (predicted == action_classes).sum().item()
        total += frames.shape[0]
        
        # 更新进度条
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.1f}%",
        })
    
    return {
        "loss": total_loss / total,
        "action_loss": total_action_loss / total,
        "mouse_loss": total_mouse_loss / total,
        "acc": 100.0 * correct / total,
    }


def eval_epoch(model, dataloader, device, class_weights=None):
    """
    评估一个 epoch
    """
    model.eval()
    total_loss = 0.0
    total_action_loss = 0.0
    total_mouse_loss = 0.0
    correct = 0
    total = 0
    
    action_criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    mouse_criterion = nn.MSELoss()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            frames = batch["frames"].to(device)
            goal_ids = batch["goal_id"].to(device)
            action_classes = batch["action_class"].to(device)
            mouse_dx = batch["mouse_dx"].to(device).unsqueeze(-1)
            mouse_dy = batch["mouse_dy"].to(device).unsqueeze(-1)
            mouse_targets = torch.cat([mouse_dx, mouse_dy], dim=-1)
            
            action_logits, mouse_pred = model(frames, goal_ids)
            
            action_loss = action_criterion(action_logits, action_classes)
            mouse_loss = mouse_criterion(mouse_pred, mouse_targets)
            loss = action_loss + 10.0 * mouse_loss
            
            total_loss += loss.item() * frames.shape[0]
            total_action_loss += action_loss.item() * frames.shape[0]
            total_mouse_loss += mouse_loss.item() * frames.shape[0]
            
            _, predicted = action_logits.max(1)
            correct += (predicted == action_classes).sum().item()
            total += frames.shape[0]
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.1f}%",
            })
    
    return {
        "loss": total_loss / total,
        "action_loss": total_action_loss / total,
        "mouse_loss": total_mouse_loss / total,
        "acc": 100.0 * correct / total,
    }


def main():
    parser = argparse.ArgumentParser(description="v5.3 双头模型训练")
    
    parser.add_argument("--data-dir", type=str, default="pathfinding_data_noidle",
                        help="过滤后数据目录")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="模型保存目录")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="批大小")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader 工作进程数")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数（用于调试）")
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", flush=True)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 数据集
    print("="*60, flush=True)
    print("加载数据", flush=True)
    print("="*60, flush=True)
    
    train_dataset = FilteredH5Dataset(
        data_dir=args.data_dir,
        frame_offsets=FRAME_OFFSETS,
        max_samples=args.max_samples,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    
    # 模型
    print("="*60, flush=True)
    print("创建模型", flush=True)
    print("="*60, flush=True)
    
    model = DualHeadModel(num_actions=3).to(device)
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # 类别权重（处理不平衡数据）
    # forward: 86.5%, turn_left: 8.3%, turn_right: 5.2%
    class_weights = torch.tensor([1.0, 5.0, 5.0])  # 提高转向类别的权重
    
    # 训练循环
    print("="*60, flush=True)
    print("开始训练", flush=True)
    print("="*60, flush=True)
    
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        print("-" * 60, flush=True)
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, class_weights
        )
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Action Loss: {train_metrics['action_loss']:.4f} | "
              f"Mouse Loss: {train_metrics['mouse_loss']:.4f} | "
              f"Acc: {train_metrics['acc']:.2f}%", flush=True)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if train_metrics["acc"] > best_acc:
            best_acc = train_metrics["acc"]
            checkpoint_path = os.path.join(args.output_dir, "goal_bc_v53_best.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ 最佳模型已保存: {checkpoint_path} (Acc: {best_acc:.2f}%)", flush=True)
        
        # 每 10 个 epoch 保存一次
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"goal_bc_v53_epoch_{epoch:03d}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Checkpoint 已保存: {checkpoint_path}", flush=True)
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, "goal_bc_v53_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ 训练完成！最终模型已保存: {final_path}", flush=True)
    print(f"最佳准确率: {best_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()
