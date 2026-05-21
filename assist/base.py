"""
L2 辅助驾驶基础模块

所有辅助功能的基类，定义统一接口
"""
import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class AssistAction:
    """辅助动作输出"""
    action_id: int          # 动作 ID（0=idle, 3=dodge, etc.）
    confidence: float       # 置信度 (0-1)
    mouse_dx: float = 0.0   # 鼠标 X 偏移
    mouse_dy: float = 0.0   # 鼠标 Y 偏移
    source: str = ""        # 来源模块名称
    priority: int = 0       # 优先级（越高越优先）


class BaseAssistModule(ABC):
    """辅助功能基类"""

    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority
        self.enabled = True
        self._last_action_time = 0
        self._cooldown_ms = 0

    @abstractmethod
    def predict(self, frame: np.ndarray, **kwargs) -> Optional[AssistAction]:
        """
        预测辅助动作

        Args:
            frame: 当前游戏画面 (H, W, 3) BGR
            **kwargs: 额外信息（血量、敌人状态等）

        Returns:
            AssistAction 或 None（不需要辅助）
        """
        pass

    def is_ready(self) -> bool:
        """检查是否可以执行（冷却时间）"""
        if not self.enabled:
            return False
        now = time.time() * 1000  # ms
        if now - self._last_action_time < self._cooldown_ms:
            return False
        return True

    def mark_executed(self):
        """标记已执行"""
        self._last_action_time = time.time() * 1000

    def reset(self):
        """重置状态"""
        self._last_action_time = 0
