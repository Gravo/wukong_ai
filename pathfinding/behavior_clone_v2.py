"""
behavior_clone.py - 行为克隆模型（v2：鼠标是输出不是输入）
从人类demo数据中学习寻路策略

架构：视觉编码器 → 分类头(动作) + 回归头(mouse_dx, mouse_dy)
鼠标移动和WASD一样是动作空间的输出，不是输入特征！

数据流：h5 → preprocess生成stacked_data.npz → 训练时mmap加载
"""
import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BC, NUM_ACTIONS, FRAME_STACK
from models.resnet_encoder import create_encoder


class DemoDataset(Dataset):
    """人类Demo数据集（预堆叠，零计算__getitem__）"""
    
    def __init__(self, data_dir, augment=True):
        self.augment = augment
        
        stacked_path = os.path.join(data_dir, "preprocessed", "stacked_data.npz")
        if not os.path.exists(stacked_path):
            print("[DemoDataset] 预堆叠文件不存在，请先运行 preprocess_data.py", flush=True)
            raise FileNotFoundError(stacked_path)
        
        print(f"[DemoDataset] 加载预堆叠数据 (mmap)...", flush=True)
        self._data = np.load(stacked_path, allow_pickle=False)
        self.frames = self._data["frames"]       # (N, 12, 224, 224) uint8
        self.actions = self._data["actions"]       # (N,) int64
        self.mouse = self._data["mouse"]           # (N, 4) float32: [dx, dy, left_btn, right_btn]
        
        total = len(self)
        print(f"[DemoDataset] 总样本数: {total}", flush=True)
        action_dist = np.bincount(self.actions, minlength=NUM_ACTIONS)
        for i, c in enumerate(action_dist):
            if c > 0:
                print(f"  动作 {i}: {c} ({c/total*100:.1f}%)")
        print(f"  鼠标dx: mean={self.mouse[:,0].mean():.3f}, std={self.mouse[:,0].std():.3f}", flush=True)
        print(f"  鼠标dy: mean={self.mouse[:,1].mean():.3f}, std={self.mouse[:,1].std():.3f}", flush=True)
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        frame = self.frames[idx].astype(np.float32) / 255.0  # (12, 224, 224)
        action = int(self.actions[idx])
        mouse_target = self.mouse[idx, :2].copy().astype(np.float32)  # 只取dx, dy作为回归目标
        
        if self.augment:
            if np.random.random() > 0.5:
                frame = frame[:, :, ::-1].copy()
                if action == 5:
                    action = 6
                elif action == 6:
                    action = 5
                mouse_target[0] = -mouse_target[0]  # 水平翻转→dx取反
        
        return frame, action, mouse_target


class BehaviorCloneModel(nn.Module):
    """行为克隆模型v2：视觉编码器 → 分类头(动作) + 回归头(dx, dy)
    
    鼠标是输出！和WASD一样是模型要预测的动作。
    """
    
    def __init__(self, latent_dim=256, num_actions=NUM_ACTIONS):
        super().__init__()
        self.encoder = create_encoder("resnet18", latent_dim=latent_dim)
        
        # 分类头：预测离散动作（idle, move_forward, move_right, ...）
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_actions),
        )
        
        # 回归头：预测鼠标dx, dy（连续值，范围[-1, 1]）
        self.mouse_regressor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 输出: [dx, dy]
        )
    
    def forward(self, x):
        feat = self.encoder(x)
        action_logits = self.classifier(feat)
        mouse_pred = self.mouse_regressor(feat)
        return action_logits, mouse_pred


def train_bc(args):
    """训练行为克隆模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BC] 设备: {device}", flush=True)
    
    dataset = DemoDataset(args.data_dir, augment=BC["augment"])
    
    dataloader = DataLoader(
        dataset,
        batch_size=BC["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    model = BehaviorCloneModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[BC] 模型参数量: {n_params:,}", flush=True)
    
    optimizer = optim.Adam(model.parameters(), lr=BC["lr"])
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    
    # 鼠标回归损失的权重（相对于分类损失）
    mouse_loss_weight = 0.5
    
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        cls_loss_sum = 0
        mse_loss_sum = 0
        correct = 0
        total = 0
        mouse_mae = 0
        
        for frames, actions, mouse_target in dataloader:
            frames = frames.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            mouse_target = mouse_target.to(device, non_blocking=True)
            
            action_logits, mouse_pred = model(frames)
            
            cls_loss = ce_criterion(action_logits, actions)
            mse_loss = mse_criterion(mouse_pred, mouse_target)
            loss = cls_loss + mouse_loss_weight * mse_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(actions)
            cls_loss_sum += cls_loss.item() * len(actions)
            mse_loss_sum += mse_loss.item() * len(actions)
            correct += (action_logits.argmax(1) == actions).sum().item()
            mouse_mae += torch.abs(mouse_pred - mouse_target).sum().item()
            total += len(actions)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        avg_cls = cls_loss_sum / total
        avg_mse = mse_loss_sum / total
        avg_mouse_mae = mouse_mae / (total * 2)
        
        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"loss={avg_loss:.4f} (cls={avg_cls:.4f} + mse={avg_mse:.4f}) | "
            f"acc={accuracy:.2%} | mouse_mae={avg_mouse_mae:.4f}",
            flush=True,
        )
        
        if accuracy > best_acc:
            best_acc = accuracy
            save_path = os.path.join(args.data_dir, "..", "checkpoints", "bc_best.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
    
    print(f"\n[BC] 训练完成！最佳准确率: {best_acc:.2%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="行为克隆训练v2（鼠标是输出）")
    parser.add_argument("--data-dir", default=BC["data_dir"])
    parser.add_argument("--epochs", type=int, default=BC["epochs"])
    
    args = parser.parse_args()
    train_bc(args)
