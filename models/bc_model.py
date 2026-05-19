"""
bc_model.py - 共享的行为克隆模型定义
用于 behavior_clone_v2.py, behavior_clone_v3.py, inference_v2.py
避免模型定义重复导致的不一致问题
"""
import torch
import torch.nn as nn
from config import NUM_ACTIONS, MODEL
from models.resnet_encoder import create_encoder


class BehaviorCloneModel(nn.Module):
    """
    行为克隆模型 v2：视觉编码器 → 分类头(动作) + 回归头(mouse_dx, mouse_dy)

    输入: (B, 12, 224, 224) 帧堆叠图像（4帧 x 3通道）
    输出:
        - action_logits: (B, NUM_ACTIONS) 动作分类 logits
        - mouse_pred: (B, 2) 鼠标移动预测 (dx, dy)
    """

    def __init__(self, latent_dim=None, num_actions=None, encoder_type=None):
        super().__init__()

        self.latent_dim = latent_dim or MODEL["latent_dim"]
        self.num_actions = num_actions or NUM_ACTIONS
        encoder_type = encoder_type or MODEL["encoder"]

        # 视觉编码器
        self.encoder = create_encoder(encoder_type, latent_dim=self.latent_dim)

        # 动作分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.num_actions),
        )

        # 鼠标回归头
        self.mouse_regressor = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: (B, 12, 224, 224) 帧堆叠图像

        Returns:
            action_logits: (B, NUM_ACTIONS)
            mouse_pred: (B, 2)
        """
        feat = self.encoder(x)
        action_logits = self.classifier(feat)
        mouse_pred = self.mouse_regressor(feat)
        return action_logits, mouse_pred

    def predict_action(self, x, deterministic=True):
        """
        预测动作（推理用）

        Args:
            x: (B, 12, 224, 224) 帧堆叠图像
            deterministic: bool，是否使用 argmax（否则采样）

        Returns:
            action: (B,) 动作 ID
            confidence: (B,) 置信度
            mouse: (B, 2) 鼠标移动
        """
        action_logits, mouse_pred = self.forward(x)

        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)

        probs = torch.softmax(action_logits, dim=-1)
        confidence = probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

        return action, confidence, mouse_pred


def create_bc_model(latent_dim=None, num_actions=None, encoder_type=None):
    """工厂函数：创建行为克隆模型"""
    return BehaviorCloneModel(
        latent_dim=latent_dim,
        num_actions=num_actions,
        encoder_type=encoder_type,
    )


def load_bc_model(model_path, device=None, **kwargs):
    """
    加载训练好的行为克隆模型

    Args:
        model_path: str，模型文件路径
        device: torch.device，加载到哪个设备
        **kwargs: 传递给 BehaviorCloneModel 的参数

    Returns:
        BehaviorCloneModel: 加载好的模型（eval 模式）
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorCloneModel(**kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
