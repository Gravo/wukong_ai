"""
resnet_encoder.py - ResNet18视觉编码器
从游戏画面中提取特征，支持预训练和微调
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from config import MODEL


class ResNetEncoder(nn.Module):
    """
    基于ResNet18的视觉编码器
    输入: (B, C, 224, 224) 帧堆叠图像
    输出: (B, latent_dim) 视觉特征向量
    """
    
    def __init__(
        self,
        latent_dim=None,
        pretrained=None,
        freeze=None,
        in_channels=None,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim or MODEL["latent_dim"]
        pretrained = pretrained if pretrained is not None else MODEL["pretrained"]
        freeze = freeze if freeze is not None else MODEL["freeze_encoder"]
        self.in_channels = in_channels or (4 * 3)  # FRAME_STACK * RGB
        
        # 加载ResNet18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = resnet18(weights=weights)
        
        # 修改第一层卷积以接受多帧堆叠输入
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            self.in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 如果使用预训练权重，将原始3通道权重复制到新通道
        if pretrained and self.in_channels > 3:
            with torch.no_grad():
                repeat_factor = self.in_channels // 3
                self.resnet.conv1.weight[:, :, :, :] = original_conv.weight.repeat(
                    1, repeat_factor, 1, 1
                ) / repeat_factor
        
        # 移除最后的全连接层，替换为新的投影头
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # 去掉fc层
        
        # 投影头
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, self.latent_dim),
            nn.ReLU(),
        )
        
        # 冻结编码器
        if freeze:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
        print("[ResNetEncoder] Encoder parameters frozen")
    
    def unfreeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = True
        print("[ResNetEncoder] Encoder parameters unfrozen")
    
    def forward(self, x):
        features = self.resnet(x)
        return self.projection(features)


class CNNEncoder(nn.Module):
    """
    轻量级CNN编码器（资源受限时的备选方案）
    4层卷积 + AdaptiveAvgPool + 全连接
    不依赖硬编码空间维度，任意输入尺寸均可工作
    """
    
    def __init__(self, latent_dim=256, in_channels=12):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.conv = nn.Sequential(
            # 224x224 -> 56x56
            nn.Conv2d(in_channels, 32, 8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # 56x56 -> 28x28
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # 28x28 -> 14x14
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # 14x14 -> 7x7
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # 自适应池化：将任意空间维度压缩到1x1，避免硬编码
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_pool(x)  # (B, 256, 1, 1)
        return self.fc(x)


def create_encoder(encoder_type=None, **kwargs):
    """编码器工厂函数"""
    encoder_type = encoder_type or MODEL["encoder"]
    
    if encoder_type == "resnet18":
        return ResNetEncoder(**kwargs)
    elif encoder_type == "cnn":
        return CNNEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
