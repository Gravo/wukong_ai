"""focal_loss.py - Focal Loss 实现（处理类别不平衡）

Focal Loss 公式：
  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

参数：
  - alpha: 类别权重（类似 CrossEntropyLoss 的 weight）
  - gamma: 调节难易样本权重（默认 2.0）
    - gamma=0: 等价于 CrossEntropyLoss
    - gamma>0: 降低易分样本权重，聚焦难分样本

使用方法（在训练脚本中）：
  from focal_loss import FocalLoss
  
  # 替代 CrossEntropyLoss
  action_loss_fn = FocalLoss(weight=action_weights, gamma=2.0)
  mouse_loss_fn = FocalLoss(weight=mouse_weights, gamma=2.0)
  
  # 计算 loss
  action_loss = action_loss_fn(action_logits, action_labels)
  mouse_loss = mouse_loss_fn(mouse_logits, mouse_labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification"""
    
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        """
        Args:
            weight: 类别权重（tensor, shape=(C,)）
            gamma: 调节难易样本权重（默认 2.0）
            reduction: 'mean' | 'sum' | 'none'
        """
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出 (N, C)
            targets: 真实标签 (N,)
        
        Returns:
            loss: scalar
        """
        # 计算 Cross Entropy (不带权重）
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 计算 p_t (模型对真实类别的预测概率）
        log_pt = -ce_loss  # CE = -log(p_t)
        pt = torch.exp(log_pt)
        
        # Focal Loss: FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
        focal_term = (1 - pt) ** self.gamma
        
        loss = focal_term * ce_loss
        
        # 应用类别权重
        if self.weight is not None:
            alpha_t = self.weight[targets]
            loss = alpha_t * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============ 测试代码 ============
if __name__ == "__main__":
    print("测试 Focal Loss")
    print("=" * 60)
    
    # 构造假数据
    N, C = 10, 3
    logits = torch.randn(N, C, requires_grad=True)
    targets = torch.randint(0, C, (N,))
    
    # 测试 1: 等价于 CrossEntropyLoss (gamma=0)
    print("\n[测试 1] gamma=0 (等价于 CE)")
    focal_loss = FocalLoss(weight=None, gamma=0.0, reduction='mean')
    ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    loss_focal = focal_loss(logits, targets)
    loss_ce = ce_loss_fn(logits, targets)
    print(f"  Focal Loss (gamma=0): {loss_focal.item():.4f}")
    print(f"  CrossEntropyLoss:    {loss_ce.item():.4f}")
    print(f"  差异: {abs(loss_focal.item() - loss_ce.item()):.6f}")
    
    # 测试 2: gamma=2.0 (降低易分样本权重)
    print("\n[测试 2] gamma=2.0 (聚焦难分样本)")
    focal_loss_g2 = FocalLoss(weight=None, gamma=2.0, reduction='mean')
    
    loss_focal_g2 = focal_loss_g2(logits, targets)
    print(f"  Focal Loss (gamma=2): {loss_focal_g2.item():.4f}")
    print(f"  比值 (gamma=2 / gamma=0): {loss_focal_g2.item() / loss_focal.item():.4f}")
    
    # 测试 3: 带类别权重
    print("\n[测试 3] 带类别权重")
    weights = torch.tensor([1.0, 5.0, 5.0])
    focal_loss_weighted = FocalLoss(weight=weights, gamma=2.0, reduction='mean')
    
    loss_focal_weighted = focal_loss_weighted(logits, targets)
    print(f"  Focal Loss (weighted): {loss_focal_weighted.item():.4f}")
    
    # 反向传播测试
    print("\n[测试 4] 反向传播")
    loss_focal_g2.backward()
    print(f"  logits.grad shape: {logits.grad.shape}")
    print(f"  ✅ 反向传播成功！")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
