"""
战斗仲裁器（Arbitrator）

灵感：自动驾驶的域控制器
- 接收各传感器（模块）的输入
- 仲裁最终控制信号
- 处理冲突和优先级

人机协同逻辑：
1. 玩家输入优先级最高（玩家按了键，AI 不覆盖）
2. 高置信度 AI 动作可以覆盖低优先级玩家动作
3. 多个 AI 模块冲突时，取优先级最高的
"""
import time
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from assist.base import BaseAssistModule, AssistAction
from config import L2_ASSIST, NUM_ACTIONS


@dataclass
class ArbitratedAction:
    """仲裁后的最终动作"""
    action_id: int          # 最终动作 ID
    mouse_dx: float = 0.0   # 鼠标 X 偏移
    mouse_dy: float = 0.0   # 鼠标 Y 偏移
    source: str = ""        # 动作来源
    confidence: float = 0.0 # 置信度
    player_overridden: bool = False  # 是否覆盖了玩家输入


class BattleArbitrator:
    """
    战斗仲裁器

    职责：
    1. 收集所有辅助模块的输出
    2. 收集玩家输入
    3. 仲裁最终动作
    4. 处理冲突和优先级
    """

    def __init__(self):
        self.modules: List[BaseAssistModule] = []
        self.config = L2_ASSIST["arbitrator"]

        # 玩家输入状态
        self.player_action: Optional[int] = None
        self.player_mouse_dx: float = 0.0
        self.player_mouse_dy: float = 0.0

        # 统计
        self.stats = {
            "total_decisions": 0,
            "player_controlled": 0,
            "ai_assisted": 0,
            "ai_overridden": 0,
        }

    def register_module(self, module: BaseAssistModule):
        """注册辅助模块"""
        self.modules.append(module)
        self.modules.sort(key=lambda m: m.priority, reverse=True)
        print(f"[Arbitrator] 注册模块: {module.name} (优先级: {module.priority})")

    def set_player_input(self, action_id: Optional[int], mouse_dx: float = 0.0, mouse_dy: float = 0.0):
        """设置玩家输入"""
        self.player_action = action_id
        self.player_mouse_dx = mouse_dx
        self.player_mouse_dy = mouse_dy

    def arbitrate(self, frame: np.ndarray, **kwargs) -> ArbitratedAction:
        """
        仲裁最终动作

        逻辑：
        1. 如果玩家有输入，优先使用玩家输入
        2. 如果 AI 模块置信度极高，可以覆盖
        3. 多个 AI 模块冲突，取优先级最高的

        Args:
            frame: 当前游戏画面

        Returns:
            ArbitratedAction: 仲裁后的动作
        """
        self.stats["total_decisions"] += 1

        # 收集所有模块的输出
        ai_actions: List[AssistAction] = []
        for module in self.modules:
            if module.is_ready():
                action = module.predict(frame, **kwargs)
                if action is not None:
                    ai_actions.append(action)

        # 情况1：玩家有输入
        if self.player_action is not None and self.player_action > 0:
            # 检查是否有高置信度 AI 动作可以覆盖
            for ai_action in ai_actions:
                if (ai_action.confidence > self.config["ai_override_threshold"]
                        and ai_action.priority > 50):
                    # AI 覆盖玩家
                    self.stats["ai_overridden"] += 1
                    ai_action.source = module.name if hasattr(module, 'name') else "ai"
                    return ArbitratedAction(
                        action_id=ai_action.action_id,
                        mouse_dx=ai_action.mouse_dx,
                        mouse_dy=ai_action.mouse_dy,
                        source=ai_action.source,
                        confidence=ai_action.confidence,
                        player_overridden=True,
                    )

            # 玩家优先
            self.stats["player_controlled"] += 1
            return ArbitratedAction(
                action_id=self.player_action,
                mouse_dx=self.player_mouse_dx,
                mouse_dy=self.player_mouse_dy,
                source="player",
                confidence=1.0,
            )

        # 情况2：玩家无输入，使用 AI 动作
        if ai_actions:
            # 取优先级最高的
            best_action = max(ai_actions, key=lambda a: a.priority)
            self.stats["ai_assisted"] += 1

            return ArbitratedAction(
                action_id=best_action.action_id,
                mouse_dx=best_action.mouse_dx,
                mouse_dy=best_action.mouse_dy,
                source=best_action.source,
                confidence=best_action.confidence,
            )

        # 情况3：无任何输入
        return ArbitratedAction(
            action_id=0,  # idle
            source="idle",
            confidence=1.0,
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.stats["total_decisions"]
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "player_pct": self.stats["player_controlled"] / total * 100,
            "ai_pct": self.stats["ai_assisted"] / total * 100,
            "override_pct": self.stats["ai_overridden"] / total * 100,
        }

    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "total_decisions": 0,
            "player_controlled": 0,
            "ai_assisted": 0,
            "ai_overridden": 0,
        }
