"""
自动面敌模块（LCC - 车道居中辅助）

灵感：自动驾驶的 LCC 系统
- 保持车辆在车道中央（保持角色面向敌人）
- 微调方向盘（微调鼠标移动）

实现方式：
1. 敌人位置检测：识别敌人在画面中的位置
2. 偏移计算：计算与画面中心的偏移
3. 鼠标修正：平滑移动鼠标修正朝向

敌人位置检测方法：
1. 锁定模式：游戏锁定敌人后，敌人始终在画面中心附近
2. 血条检测：Boss 血条位置指示敌人方向
3. 视觉检测：用目标检测模型识别敌人位置
"""
import numpy as np
import cv2
from typing import Optional, Tuple

from assist.base import BaseAssistModule, AssistAction
from config import BLOOD_REGION, L2_ASSIST


class AutoFaceModule(BaseAssistModule):
    """
    自动面敌模块

    工作流程：
    1. 检测敌人在画面中的位置
    2. 计算与画面中心的偏移
    3. 生成鼠标修正动作
    """

    def __init__(self):
        super().__init__(name="auto_face", priority=80)

        # 配置
        config = L2_ASSIST["auto_face"]
        self.deadzone_px = config["deadzone_px"]
        self.sensitivity = config["sensitivity"]
        self.smoothing = config["smoothing"]

        # 状态
        self.prev_mouse_dx = 0.0
        self.prev_mouse_dy = 0.0
        self.enemy_position = None  # (x, y) 敌人位置

    def detect_enemy_position(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        检测敌人在画面中的位置

        方法1：检测 Boss 血条位置（最简单）
        方法2：检测锁定标记（如果已锁定）
        方法3：视觉检测（需要训练模型）
        """
        h, w = frame.shape[:2]

        # 方法1：检测 Boss 血条
        boss_region = BLOOD_REGION["boss_hp"]
        x, y, rw, rh = (
            boss_region["left"], boss_region["top"],
            boss_region["width"], boss_region["height"]
        )

        # 确保坐标在画面范围内
        x = min(x, w - 1)
        y = min(y, h - 1)
        rw = min(rw, w - x)
        rh = min(rh, h - y)

        # 裁剪血条区域
        blood_bar = frame[y:y + rh, x:x + rw]

        if blood_bar.size == 0:
            return None

        # 转 HSV 检测红色
        hsv = cv2.cvtColor(blood_bar, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 80, 80])
        upper = np.array([15, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # 计算红色区域的中心
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"]) + x
            cy = int(moments["m01"] / moments["m00"]) + y
            return (cx, cy)

        # 如果有血条但检测不到红色，返回血条中心
        if rw > 0 and rh > 0:
            return (x + rw // 2, y + rh // 2)

        return None

    def calculate_offset(self, enemy_pos: Tuple[int, int], frame_shape: Tuple[int, int]) -> Tuple[float, float]:
        """
        计算敌人与画面中心的偏移

        Returns:
            (dx, dy): 偏移量（像素），正值表示敌人在右边/下边
        """
        h, w = frame_shape[:2]
        center_x = w // 2
        center_y = h // 2

        dx = enemy_pos[0] - center_x
        dy = enemy_pos[1] - center_y

        return (dx, dy)

    def predict(self, frame: np.ndarray, **kwargs) -> Optional[AssistAction]:
        """
        预测面敌修正动作

        Args:
            frame: 当前游戏画面 (H, W, 3) BGR

        Returns:
            AssistAction 如果需要修正，否则 None
        """
        # 检测敌人位置
        enemy_pos = self.detect_enemy_position(frame)

        if enemy_pos is None:
            self.enemy_position = None
            return None

        self.enemy_position = enemy_pos

        # 计算偏移
        dx, dy = self.calculate_offset(enemy_pos, frame.shape)

        # 检查是否在死区内
        if abs(dx) < self.deadzone_px and abs(dy) < self.deadzone_px:
            return None

        # 计算鼠标移动量（归一化到 [-1, 1]）
        h, w = frame.shape[:2]
        mouse_dx = (dx / (w // 2)) * self.sensitivity
        mouse_dy = (dy / (h // 2)) * self.sensitivity

        # 平滑处理
        mouse_dx = self.smoothing * self.prev_mouse_dx + (1 - self.smoothing) * mouse_dx
        mouse_dy = self.smoothing * self.prev_mouse_dy + (1 - self.smoothing) * mouse_dy

        self.prev_mouse_dx = mouse_dx
        self.prev_mouse_dy = mouse_dy

        # 计算置信度（偏移越大，置信度越高）
        distance = np.sqrt(dx ** 2 + dy ** 2)
        max_distance = np.sqrt((w // 2) ** 2 + (h // 2) ** 2)
        confidence = min(1.0, distance / max_distance)

        return AssistAction(
            action_id=0,  # idle（不执行动作，只修正朝向）
            confidence=confidence,
            mouse_dx=mouse_dx,
            mouse_dy=mouse_dy,
            source="auto_face",
            priority=self.priority,
        )

    def get_enemy_position(self) -> Optional[Tuple[int, int]]:
        """获取当前敌人位置"""
        return self.enemy_position
