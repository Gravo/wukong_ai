"""
L2 辅助驾驶集成示例

展示如何将 L2 辅助系统集成到现有的推理流程中

场景：
1. 使用 BC 模型进行寻路（长距离移动）
2. 使用 L2 辅助进行战斗（短距离反应）

这种组合实现了"分层决策"：
- 高层：BC 模型决定去哪里（寻路）
- 低层：L2 辅助决定怎么打（战斗）
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assist.runner import L2AssistRunner
from assist.arbitrator import ArbitratedAction
from env.screen_capture import ScreenCapture
from env.action_executor import ActionExecutor


class HybridController:
    """
    混合控制器

    结合 BC 模型（寻路）和 L2 辅助（战斗）

    工作流程：
    1. 检测是否有敌人（Boss 血条是否出现）
    2. 如果有敌人：使用 L2 辅助战斗
    3. 如果无敌人：使用 BC 模型寻路
    """

    def __init__(self, bc_model_path: str = None, dodge_model_path: str = None):
        """
        初始化混合控制器

        Args:
            bc_model_path: BC 模型路径（用于寻路）
            dodge_model_path: 闪避模型路径（用于战斗）
        """
        # L2 辅助系统
        self.l2_runner = L2AssistRunner(dodge_model_path=dodge_model_path)

        # BC 模型（如果提供）
        self.bc_model = None
        if bc_model_path and os.path.exists(bc_model_path):
            self._load_bc_model(bc_model_path)

        # 状态
        self.mode = "exploration"  # "exploration" 或 "combat"
        self.enemy_detected = False

        # 血量检测
        from env.blood_detector import BloodDetector
        self.blood_detector = BloodDetector()

        print("[Hybrid] 混合控制器初始化完成")
        print(f"  BC 模型: {'已加载' if self.bc_model else '未加载'}")
        print(f"  L2 辅助: 已启用")

    def _load_bc_model(self, model_path: str):
        """加载 BC 模型"""
        try:
            from models.bc_model import load_bc_model
            self.bc_model = load_bc_model(model_path)
            print(f"[Hybrid] BC 模型加载成功: {model_path}")
        except Exception as e:
            print(f"[Hybrid] BC 模型加载失败: {e}")

    def detect_enemy(self, frame: np.ndarray) -> bool:
        """
        检测是否有敌人

        方法：检测 Boss 血条是否出现
        """
        vitals = self.blood_detector.get_all_vitals(frame)
        boss_hp = vitals["boss_hp"]

        # 如果 Boss 血条有血量，说明有敌人
        return boss_hp > 0.02

    def get_bc_action(self, frame: np.ndarray) -> int:
        """
        获取 BC 模型的寻路动作

        Args:
            frame: 当前游戏画面

        Returns:
            action_id: 动作 ID
        """
        if self.bc_model is None:
            return 4  # 默认前进

        try:
            # 预处理帧
            import cv2
            small = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)

            # 推理
            with torch.no_grad():
                action_logits, _ = self.bc_model(frame_tensor)
                action = action_logits.argmax(1).item()

            return action
        except Exception as e:
            print(f"[Hybrid] BC 推理失败: {e}")
            return 4  # 默认前进

    def step(self, frame: np.ndarray,
             player_action: int = None,
             player_mouse_dx: float = 0.0,
             player_mouse_dy: float = 0.0) -> ArbitratedAction:
        """
        执行一步决策

        Args:
            frame: 当前游戏画面
            player_action: 玩家动作
            player_mouse_dx: 玩家鼠标 X
            player_mouse_dy: 玩家鼠标 Y

        Returns:
            ArbitratedAction: 最终动作
        """
        # 检测是否有敌人
        self.enemy_detected = self.detect_enemy(frame)

        # 根据模式选择动作
        if self.enemy_detected:
            # 战斗模式：使用 L2 辅助
            if self.mode != "combat":
                print("[Hybrid] 检测到敌人，切换到战斗模式")
                self.mode = "combat"

            action = self.l2_runner.step(
                frame,
                player_action=player_action,
                player_mouse_dx=player_mouse_dx,
                player_mouse_dy=player_mouse_dy,
            )
        else:
            # 探索模式：使用 BC 模型
            if self.mode != "exploration":
                print("[Hybrid] 敌人消失，切换到探索模式")
                self.mode = "exploration"

            bc_action = self.get_bc_action(frame)

            # 如果玩家有输入，玩家优先
            if player_action is not None and player_action > 0:
                final_action = player_action
                source = "player"
            else:
                final_action = bc_action
                source = "bc_model"

            action = ArbitratedAction(
                action_id=final_action,
                mouse_dx=player_mouse_dx,
                mouse_dy=player_mouse_dy,
                source=source,
                confidence=1.0,
            )

        return action


def main():
    """运行混合控制器示例"""
    import argparse

    parser = argparse.ArgumentParser(description="混合控制器示例")
    parser.add_argument("--bc-model", default=None, help="BC 模型路径")
    parser.add_argument("--dodge-model", default=None, help="闪避模型路径")
    parser.add_argument("--duration", type=int, default=60, help="运行时长（秒）")

    args = parser.parse_args()

    # 创建混合控制器
    controller = HybridController(
        bc_model_path=args.bc_model,
        dodge_model_path=args.dodge_model,
    )

    # 创建截图器和执行器
    capture = ScreenCapture()
    executor = ActionExecutor()

    print(f"\n[示例] 开始运行 ({args.duration}秒)")
    print("[示例] 按 Ctrl+C 停止\n")

    start_time = time.time()
    frame_count = 0

    try:
        while time.time() - start_time < args.duration:
            # 截图
            frame = capture.grab()
            if frame is None:
                time.sleep(0.01)
                continue

            # 获取动作
            action = controller.step(frame)

            # 执行动作
            if action.action_id > 0:
                executor.execute(action.action_id)

            # 执行鼠标修正
            if action.mouse_dx != 0 or action.mouse_dy != 0:
                import pydirectinput
                dx = int(action.mouse_dx * 100)
                dy = int(action.mouse_dy * 100)
                if abs(dx) > 1 or abs(dy) > 1:
                    pydirectinput.moveRel(dx, dy, relative=True)

            frame_count += 1

            # 打印状态
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"[{elapsed:.0f}s] 模式: {controller.mode} | "
                      f"动作: {action.source} | "
                      f"FPS: {fps:.1f}")

            # 控制帧率
            time.sleep(0.1)  # 10 FPS

    except KeyboardInterrupt:
        print("\n[示例] 用户中断")

    finally:
        # 清理
        capture.release()
        executor.release_all()

        print(f"\n[示例] 运行结束")
        print(f"  总帧数: {frame_count}")
        print(f"  运行时间: {time.time() - start_time:.1f}秒")


if __name__ == "__main__":
    import torch
    main()
