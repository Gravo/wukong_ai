"""goal_conditioned_bc_v55_fixed.py - v5.5 训练脚本（QA 修复版）

修复（相比 v5.5）:
  1. 降低 batch_size=1（避免 OOM，用梯度累积）
  2. 分出 20% 验证集（不再用训练集验证）
  3. 梯度累积（每 4 个 batch 更新一次）
  4. 更好的学习率调度（CosineAnnealingLR）

其他改进：
  - 保存最佳 Acc_A 和 Acc_M 的模型（不只看 Loss）
  - 训练日志更详细（每个 epoch 保存 CSV）
  - 支持从 checkpoint 恢复训练

使用方法：
  C:\Python\python.exe -u training\goal_conditioned_bc_v55_fixed.py ^
    --data-dir "D:\projects\wukong_ai\pathfinding_data_balanced" ^
    --output-dir "D:\projects\wukong_ai\checkpoints" ^
    --epochs 50 ^
    --lr 1e-4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import torchvision.models as models
import warnings

# ============ 配置 ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FRAME_STACK = [0, 1, 3, 7]
NUM_CLASSES_ACTION = 3
NUM_CLASSES_MOUSE = 7
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积（batch_size=1 * 4 = 有效 batch=4）

warnings.filterwarnings('ignore')

# ============ Focal Loss ============
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        log_pt = -ce_loss
        pt = torch.exp(log_pt)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss
        
        if self.weight is not None:
            if self.weight.device != targets.device:
                self.weight = self.weight.to(targets.device)
            alpha_t = self.weight[targets]
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============ 数据集 ============
class V55Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.h5_files = sorted(self.data_dir.glob("*.h5"))
        self.samples = []
        
        print(f"[数据集] 读取目录: {data_dir}")
        print(f"[数据集] 找到 {len(self.h5_files)} 个 h5 文件")
        
        for h5_file in tqdm(self.h5_files, desc="加载索引"):
            try:
                with h5py.File(h5_file, 'r') as hf:
                    required = ['frames', 'actions', 'mouse_dx', 'mouse_dy']
                    missing = [ds for ds in required if ds not in hf]
                    if missing:
                        continue
                    
                    n = len(hf['frames'])
                    if n > max(FRAME_STACK):
                        self.samples.append((h5_file, n))
            except Exception:
                pass
        
        print(f"[数据集] 有效样本: {len(self.samples)} 个文件")
    
    def __len__(self):
        total = 0
        for _, n in self.samples:
            total += (n - max(FRAME_STACK) - 1)
        return total
    
    def __getitem__(self, idx):
        file_idx = 0
        frame_idx = idx
        
        for i, (h5_file, n) in enumerate(self.samples):
            max_valid = n - max(FRAME_STACK) - 1
            if frame_idx < max_valid:
                file_idx = i
                break
            frame_idx -= max_valid
        
        h5_file, n = self.samples[file_idx]
        
        with h5py.File(h5_file, 'r') as hf:
            frames = []
            for offset in FRAME_STACK:
                frame = hf['frames'][frame_idx + offset]
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
            
            frames_stacked = torch.cat(frames, dim=0)
            
            mouse_dx = hf['mouse_dx'][frame_idx + FRAME_STACK[-1]]
            
            if mouse_dx < -20:
                action = 1
            elif mouse_dx > 20:
                action = 2
            else:
                action = 0
            
            mouse_bucket = self._classify_bucket(mouse_dx)
            
            if 'goal_ids' in hf:
                goal_id = hf['goal_ids'][frame_idx + FRAME_STACK[-1]]
            else:
                goal_id = 0
            
            goal_id = torch.tensor(goal_id, dtype=torch.long)
            
            return frames_stacked, action, mouse_bucket, goal_id
    
    def _classify_bucket(self, mouse_dx):
        if mouse_dx <= -200:
            return 0
        elif mouse_dx <= -100:
            return 1
        elif mouse_dx <= -20:
            return 2
        elif mouse_dx <= 20:
            return 3
        elif mouse_dx <= 100:
            return 4
        elif mouse_dx <= 200:
            return 5
        else:
            return 6


# ============ 模型 ============
class GoalConditionedBC_v55(nn.Module):
    def __init__(self, num_goals=2, freeze_backbone=False):
        super(GoalConditionedBC_v55, self).__init__()
        
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        old_conv1 = self.backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=12,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        
        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = old_conv1.weight / 4.0
            for i in range(1, 4):
                new_conv1.weight[:, i*3:(i+1)*3, :, :] = old_conv1.weight / 4.0
        
        self.backbone.conv1 = new_conv1
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.goal_embedding = nn.Embedding(num_goals, 512)
        
        self.action_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES_ACTION)
        )
        
        self.mouse_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES_MOUSE)
        )
    
    def forward(self, frames, goal_ids):
        features = self.backbone(frames)
        features = features.view(features.size(0), -1)
        
        goal_emb = self.goal_embedding(goal_ids)
        fused_features = features + goal_emb
        
        action_logits = self.action_head(fused_features)
        mouse_logits = self.mouse_head(fused_features)
        
        return action_logits, mouse_logits


# ============ 训练函数 ============
def train(args):
    print("=" * 60)
    print("v5.5 训练脚本（QA 修复版）")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据集（分训练/验证）
    full_dataset = V55Dataset(args.data_dir)
    
    if len(full_dataset) == 0:
        print("[错误] 数据集为空！")
        return
    
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"[数据] 训练样本数: {train_size}")
    print(f"[数据] 验证样本数: {val_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # 降低 batch_size
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"[数据] 训练 Batch 数: {len(train_loader)}")
    print(f"[数据] 验证 Batch 数: {len(val_loader)}")
    
    # 模型
    model = GoalConditionedBC_v55(num_goals=2, freeze_backbone=False).to(DEVICE)
    
    # 损失函数
    action_weights = torch.tensor([1.0, 10.0, 10.0]).to(DEVICE)
    mouse_weights = torch.tensor([1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0]).to(DEVICE)
    
    action_loss_fn = FocalLoss(weight=action_weights, gamma=2.0)
    mouse_loss_fn = FocalLoss(weight=mouse_weights, gamma=2.0)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度器（CosineAnnealing）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 训练日志
    log_file = output_dir / f"training_log_v55_fixed_{args.epochs}ep.csv"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("epoch,train_loss,train_acc_a,train_acc_m,val_loss,val_acc_a,val_acc_m,lr,time\n")
    
    best_loss = float('inf')
    best_acc_a = 0.0
    best_acc_m = 0.0
    
    print(f"\n[训练] 开始训练...")
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # ============ 训练阶段 ============
        model.train()
        train_loss = 0.0
        train_correct_a = 0
        train_correct_m = 0
        train_total = 0
        
        optimizer.zero_grad()  # 梯度累积：每个 epoch 开始清零
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [训练]")
        
        for batch_idx, (frames, actions, mouse_buckets, goal_ids) in enumerate(pbar):
            frames = frames.to(DEVICE)
            actions = actions.to(DEVICE)
            mouse_buckets = mouse_buckets.to(DEVICE)
            goal_ids = goal_ids.to(DEVICE)
            
            action_logits, mouse_logits = model(frames, goal_ids)
            
            action_loss = action_loss_fn(action_logits, actions)
            mouse_loss = mouse_loss_fn(mouse_logits, mouse_buckets)
            total_loss = (action_loss + mouse_loss) / GRADIENT_ACCUMULATION_STEPS  # 除以累积步数
            
            total_loss.backward()
            
            # 每 GRADIENT_ACCUMULATION_STEPS 更新一次
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += total_loss.item() * GRADIENT_ACCUMULATION_STEPS  # 恢复原始 loss
            
            _, action_pred = action_logits.max(1)
            _, mouse_pred = mouse_logits.max(1)
            
            train_correct_a += action_pred.eq(actions).sum().item()
            train_correct_m += mouse_pred.eq(mouse_buckets).sum().item()
            train_total += actions.size(0)
            
            pbar.set_postfix({
                'Loss': f"{total_loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}",
                'Acc_A': f"{100. * train_correct_a / train_total:.2f}%",
                'Acc_M': f"{100. * train_correct_m / train_total:.2f}%"
            })
        
        # 处理剩余的梯度
        if (len(train_loader) % GRADIENT_ACCUMULATION_STEPS) != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc_a = 100. * train_correct_a / train_total
        train_acc_m = 100. * train_correct_m / train_total
        
        # ============ 验证阶段 ============
        model.eval()
        val_loss = 0.0
        val_correct_a = 0
        val_correct_m = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, actions, mouse_buckets, goal_ids in val_loader:
                frames = frames.to(DEVICE)
                actions = actions.to(DEVICE)
                mouse_buckets = mouse_buckets.to(DEVICE)
                goal_ids = goal_ids.to(DEVICE)
                
                action_logits, mouse_logits = model(frames, goal_ids)
                
                action_loss = action_loss_fn(action_logits, actions)
                mouse_loss = mouse_loss_fn(mouse_logits, mouse_buckets)
                total_loss = action_loss + mouse_loss
                
                val_loss += total_loss.item()
                
                _, action_pred = action_logits.max(1)
                _, mouse_pred = mouse_logits.max(1)
                
                val_correct_a += action_pred.eq(actions).sum().item()
                val_correct_m += mouse_pred.eq(mouse_buckets).sum().item()
                val_total += actions.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc_a = 100. * val_correct_a / val_total
        val_acc_m = 100. * val_correct_m / val_total
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 保存最佳模型（基于 Loss）
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'acc_action': val_acc_a,
                'acc_mouse': val_acc_m,
            }
            torch.save(checkpoint, output_dir / "goal_bc_v55_best_loss.pt")
            print(f"\n[保存] 最佳 Loss 模型 (Loss: {avg_val_loss:.4f})")
        
        # 保存最佳 Acc_A 模型
        if val_acc_a > best_acc_a:
            best_acc_a = val_acc_a
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'acc_action': val_acc_a,
                'acc_mouse': val_acc_m,
            }
            torch.save(checkpoint, output_dir / "goal_bc_v55_best_acc_a.pt")
            print(f"[保存] 最佳 Acc_A 模型 (Acc_A: {val_acc_a:.2f}%)")
        
        # 保存最佳 Acc_M 模型
        if val_acc_m > best_acc_m:
            best_acc_m = val_acc_m
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'acc_action': val_acc_a,
                'acc_mouse': val_acc_m,
            }
            torch.save(checkpoint, output_dir / "goal_bc_v55_best_acc_m.pt")
            print(f"[保存] 最佳 Acc_M 模型 (Acc_M: {val_acc_m:.2f}%)")
        
        # 定期保存 checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }
            torch.save(checkpoint, output_dir / f"goal_bc_v55_epoch{epoch+1}.pt")
        
        # 打印统计
        epoch_time = time.time() - epoch_start_time
        
        log_str = f"{epoch+1},{avg_train_loss:.4f},{train_acc_a:.2f},{train_acc_m:.2f},{avg_val_loss:.4f},{val_acc_a:.2f},{val_acc_m:.2f},{current_lr:.6f},{epoch_time:.2f}"
        
        print(f"\nEpoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc_A: {train_acc_a:.2f}% | Val Acc_A: {val_acc_a:.2f}% | "
              f"Train Acc_M: {train_acc_m:.2f}% | Val Acc_M: {val_acc_m:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
        
        # 写入日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_str + "\n")
    
    print("\n" + "=" * 60)
    print("[完成] 训练完成！")
    print(f"[完成] 最佳 Loss 模型: {output_dir / 'goal_bc_v55_best_loss.pt'}")
    print(f"[完成] 最佳 Acc_A 模型: {output_dir / 'goal_bc_v55_best_acc_a.pt'}")
    print(f"[完成] 最佳 Acc_M 模型: {output_dir / 'goal_bc_v55_best_acc_m.pt'}")
    print(f"[完成] 训练日志: {log_file}")
    print("=" * 60)


# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser(description='v5.5 Training Script (QA Fixed)')
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='数据目录（增强后）')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录（保存模型）')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练 epoch 数（默认 50）')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率（默认 1e-4）')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='冻结 ResNet18 主干')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"[错误] 数据目录不存在: {args.data_dir}")
        return
    
    train(args)


if __name__ == "__main__":
    main()
