"""
behavior_clone.py - 行为克隆模型
从人类demo数据中学习寻路策略
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BC, NUM_ACTIONS, FRAME_STACK
from models.resnet_encoder import create_encoder


class DemoDataset(Dataset):
    """人类Demo数据集"""
    
    def __init__(self, data_dir, frame_stack=4, augment=True):
        import h5py
        
        self.frame_stack = frame_stack
        self.augment = augment
        self.frames = []
        self.actions = []
        
        # 加载所有h5文件
        h5_files = glob.glob(os.path.join(data_dir, "*.h5"))
        print(f"[DemoDataset] 找到 {len(h5_files)} 个数据文件")
        
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                frames = f["frames"][:]   # (N, 224, 224, 3)
                actions = f["actions"][:]  # (N,)
                
                # 构建帧堆叠样本
                for i in range(frame_stack - 1, len(frames)):
                    stacked = frames[i - frame_stack + 1 : i + 1]  # (stack, H, W, 3)
                    # 转为 (stack*3, H, W)
                    stacked = np.concatenate(
                        [f.transpose(2, 0, 1) for f in stacked], axis=0
                    )
                    self.frames.append(stacked.astype(np.float32) / 255.0)
                    self.actions.append(actions[i])
        
        self.frames = np.array(self.frames, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.int64)
        
        print(f"[DemoDataset] 总样本数: {len(self)}")
        action_dist = np.bincount(self.actions, minlength=NUM_ACTIONS)
        for i, c in enumerate(action_dist):
            if c > 0:
                print(f"  动作 {i}: {c} ({c/len(self)*100:.1f}%)")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        action = self.actions[idx]
        
        if self.augment:
            # 简单数据增强：随机水平翻转
            if np.random.random() > 0.5:
                frame = frame[:, :, ::-1].copy()
                # 翻转时交换左右移动动作
                if action == 5:  # move_right
                    action = 6  # move_left
                elif action == 6:  # move_left
                    action = 5  # move_right
        
        return frame, action


class BehaviorCloneModel(nn.Module):
    """行为克隆模型：视觉编码器 + 分类头"""
    
    def __init__(self, latent_dim=256, num_actions=NUM_ACTIONS):
        super().__init__()
        self.encoder = create_encoder("resnet18", latent_dim=latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )
    
    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat)


def train_bc(args):
    """训练行为克隆模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BC] 设备: {device}")
    
    # 数据集
    dataset = DemoDataset(
        args.data_dir,
        frame_stack=FRAME_STACK,
        augment=BC["augment"],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=BC["batch_size"],
        shuffle=True,
        num_workers=0,  # Windows compatibility: 0 avoids multiprocessing issues
        pin_memory=True,
    )
    
    # 模型
    model = BehaviorCloneModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=BC["lr"])
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    best_acc = 0
    for epoch in range(BC["epochs"]):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for frames, actions in dataloader:
            frames = frames.to(device)
            actions = actions.to(device)
            
            logits = model(frames)
            loss = criterion(logits, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(actions)
            correct += (logits.argmax(1) == actions).sum().item()
            total += len(actions)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        print(f"Epoch {epoch+1}/{BC['epochs']}: loss={avg_loss:.4f}, acc={accuracy:.2%}")
        
        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            save_path = os.path.join(args.data_dir, "..", "checkpoints", "bc_best.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
    
    print(f"\n[BC] 训练完成！最佳准确率: {best_acc:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="行为克隆训练")
    parser.add_argument("--data-dir", default=BC["data_dir"])
    parser.add_argument("--epochs", type=int, default=BC["epochs"])
    
    args = parser.parse_args()
    train_bc(args)
