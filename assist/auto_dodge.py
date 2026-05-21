"""
自动闪避模块（AEB - 自动紧急制动）

灵感：自动驾驶的 AEB 系统
- 检测前方障碍（敌人攻击前摇）
- 紧急制动（自动闪避）

实现方式：
1. 视觉检测：识别敌人攻击动画
2. 置信度评估：判断是否真的在攻击
3. 时机控制：在最佳时机触发闪避

数据采集：
- 录制敌人攻击前摇的画面
- 标记闪避时机
- 训练二分类模型（攻击/非攻击）
"""
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Optional

from assist.base import BaseAssistModule, AssistAction


class AttackDetector(nn.Module):
    """
    攻击前摇检测器

    输入: 4帧堆叠画面 (12, 224, 224)
    输出: 攻击概率 (0-1)
    """

    def __init__(self, in_channels=12):
        super().__init__()
        # 轻量级 CNN，适合实时检测
        self.features = nn.Sequential(
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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class AutoDodgeModule(BaseAssistModule):
    """
    自动闪避模块

    工作流程：
    1. 接收连续帧画面
    2. 检测敌人攻击前摇
    3. 如果检测到攻击且置信度足够，触发闪避
    """

    def __init__(self, model_path: Optional[str] = None, device=None):
        super().__init__(name="auto_dodge", priority=100)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttackDetector().to(self.device)

        if model_path:
            self.load_model(model_path)

        # 帧缓冲区
        self.frame_buffer = []
        self.buffer_size = 4  # 需要 4 帧堆叠

        # 检测参数
        self.confidence_threshold = 0.85
        self.cooldown_ms = 500

        # 状态
        self._last_dodge_time = 0
        self._attack_detected = False

    def load_model(self, model_path: str):
        """加载训练好的模型"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"[AutoDodge] 模型加载成功: {model_path}")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理单帧"""
        # 缩放到 224x224
        resized = cv2.resize(frame, (224, 224))
        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # 归一化
        normalized = rgb.astype(np.float32) / 255.0
        # (H, W, 3) -> (3, H, W)
        return normalized.transpose(2, 0, 1)

    def add_frame(self, frame: np.ndarray):
        """添加帧到缓冲区"""
        processed = self.preprocess_frame(frame)
        self.frame_buffer.append(processed)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

    def predict(self, frame: np.ndarray, **kwargs) -> Optional[AssistAction]:
        """
        检测攻击并预测闪避动作

        Args:
            frame: 当前游戏画面 (H, W, 3) BGR

        Returns:
            AssistAction 如果需要闪避，否则 None
        """
        # 添加帧到缓冲区
        self.add_frame(frame)

        # 检查是否可以执行（冷却时间）
        now = time.time() * 1000
        if now - self._last_dodge_time < self.cooldown_ms:
            return None

        # 检查缓冲区是否满
        if len(self.frame_buffer) < self.buffer_size:
            return None

        # 帧堆叠
        stacked = np.concatenate(self.frame_buffer, axis=0)
        frame_tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device)

        # 检测攻击
        with torch.no_grad():
            attack_prob = self.model(frame_tensor).item()

        # 如果检测到攻击且置信度足够
        if attack_prob > self.confidence_threshold:
            self._last_dodge_time = now
            self._attack_detected = True

            return AssistAction(
                action_id=3,  # dodge
                confidence=attack_prob,
                source="auto_dodge",
                priority=self.priority,
            )

        self._attack_detected = False
        return None

    def is_attack_detected(self) -> bool:
        """是否检测到攻击"""
        return self._attack_detected


class AutoDodgeRuleBased(BaseAssistModule):
    """
    基于规则的自动闪避（无需训练）

    使用简单的图像处理检测攻击：
    1. 检测画面中的快速运动（光流）
    2. 检测敌人动作幅度
    3. 检测攻击特效（红色闪光等）
    """

    def __init__(self):
        super().__init__(name="auto_dodge_rule", priority=50)
        self.prev_gray = None
        self.motion_threshold = 30  # 运动阈值
        self.cooldown_ms = 500

    def predict(self, frame: np.ndarray, **kwargs) -> Optional[AssistAction]:
        """基于规则检测攻击"""
        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        # 计算帧差
        diff = cv2.absdiff(self.prev_gray, gray)
        motion = np.mean(diff)

        self.prev_gray = gray

        # 检查冷却
        now = time.time() * 1000
        if now - self._last_action_time < self.cooldown_ms:
            return None

        # 如果运动幅度超过阈值
        if motion > self.motion_threshold:
            self._last_action_time = now

            # 计算置信度（归一化）
            confidence = min(1.0, motion / 100.0)

            return AssistAction(
                action_id=3,  # dodge
                confidence=confidence,
                source="auto_dodge_rule",
                priority=self.priority,
            )

        return None
