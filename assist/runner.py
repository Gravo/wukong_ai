"""
L2 辅助驾驶运行器

整合所有辅助模块，提供统一的运行接口

使用方式：
    # 创建运行器
    runner = L2AssistRunner()

    # 游戏循环
    while True:
        frame = capture.grab()
        player_input = get_player_input()

        # 获取辅助动作
        action = runner.step(frame, player_input)

        # 执行动作
        executor.execute(action.action_id)
        if action.mouse_dx != 0 or action.mouse_dy != 0:
            executor.move_mouse(action.mouse_dx, action.mouse_dy)
"""
import time
import numpy as np
from typing import Optional, Dict, Any

from assist.arbitrator import BattleArbitrator, ArbitratedAction
from assist.auto_dodge import AutoDodgeModule, AutoDodgeRuleBased
from assist.auto_face import AutoFaceModule
from config import L2_ASSIST


class L2AssistRunner:
    """
    L2 辅助驾驶运行器

    功能：
    1. 管理所有辅助模块
    2. 提供统一的 step() 接口
    3. 处理模块启停
    4. 记录统计信息
    """

    def __init__(self, dodge_model_path: Optional[str] = None):
        """
        初始化 L2 辅助系统

        Args:
            dodge_model_path: 自动闪避模型路径（None 则使用规则方法）
        """
        self.config = L2_ASSIST
        self.enabled = self.config["enabled"]

        # 创建仲裁器
        self.arbitrator = BattleArbitrator()

        # 注册辅助模块
        self._init_modules(dodge_model_path)

        # 状态
        self._step_count = 0
        self._start_time = time.time()

        print(f"[L2Assist] 初始化完成")
        print(f"  自动闪避: {'开启' if self.config['auto_dodge']['enabled'] else '关闭'}")
        print(f"  自动面敌: {'开启' if self.config['auto_face']['enabled'] else '关闭'}")
        print(f"  自动连招: {'开启' if self.config['auto_combo']['enabled'] else '关闭'}")

    def _init_modules(self, dodge_model_path: Optional[str]):
        """初始化辅助模块"""
        # 自动闪避（AEB）
        if self.config["auto_dodge"]["enabled"]:
            if dodge_model_path:
                # 使用训练好的模型
                dodge_module = AutoDodgeModule(model_path=dodge_model_path)
            else:
                # 使用规则方法
                dodge_module = AutoDodgeRuleBased()
                print("[L2Assist] 自动闪避使用规则方法（无需训练）")

            dodge_module.confidence_threshold = self.config["auto_dodge"]["confidence_threshold"]
            dodge_module.cooldown_ms = self.config["auto_dodge"]["cooldown_ms"]
            self.arbitrator.register_module(dodge_module)

        # 自动面敌（LCC）
        if self.config["auto_face"]["enabled"]:
            face_module = AutoFaceModule()
            self.arbitrator.register_module(face_module)

    def step(self, frame: np.ndarray,
             player_action: Optional[int] = None,
             player_mouse_dx: float = 0.0,
             player_mouse_dy: float = 0.0,
             **kwargs) -> ArbitratedAction:
        """
        执行一步辅助决策

        Args:
            frame: 当前游戏画面 (H, W, 3) BGR
            player_action: 玩家当前动作（None 表示无输入）
            player_mouse_dx: 玩家鼠标 X 偏移
            player_mouse_dy: 玩家鼠标 Y 偏移
            **kwargs: 额外信息

        Returns:
            ArbitratedAction: 仲裁后的动作
        """
        if not self.enabled:
            return ArbitratedAction(action_id=player_action or 0, source="disabled")

        self._step_count += 1

        # 设置玩家输入
        self.arbitrator.set_player_input(player_action, player_mouse_dx, player_mouse_dy)

        # 仲裁动作
        action = self.arbitrator.arbitrate(frame, **kwargs)

        return action

    def enable(self):
        """启用辅助"""
        self.enabled = True
        print("[L2Assist] 辅助系统已启用")

    def disable(self):
        """禁用辅助"""
        self.enabled = False
        print("[L2Assist] 辅助系统已禁用")

    def toggle_module(self, module_name: str):
        """切换模块启停"""
        for module in self.arbitrator.modules:
            if module.name == module_name:
                module.enabled = not module.enabled
                status = "启用" if module.enabled else "禁用"
                print(f"[L2Assist] 模块 {module_name} 已{status}")
                return
        print(f"[L2Assist] 未找到模块: {module_name}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.arbitrator.get_stats()
        stats["step_count"] = self._step_count
        stats["uptime_seconds"] = time.time() - self._start_time
        return stats

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print(f"\n[L2Assist] 统计信息:")
        print(f"  总决策数: {stats['total_decisions']}")
        print(f"  玩家控制: {stats.get('player_pct', 0):.1f}%")
        print(f"  AI 辅助: {stats.get('ai_pct', 0):.1f}%")
        print(f"  AI 覆盖: {stats.get('override_pct', 0):.1f}%")
        print(f"  运行时间: {stats['uptime_seconds']:.1f}秒")


class L2AssistDemo:
    """
    L2 辅助驾驶演示模式

    用于测试和展示辅助功能
    """

    def __init__(self):
        self.runner = L2AssistRunner()

    def run(self, duration: int = 60):
        """
        运行演示

        Args:
            duration: 演示时长（秒）
        """
        from env.screen_capture import ScreenCapture
        from env.action_executor import ActionExecutor

        capture = ScreenCapture()
        executor = ActionExecutor()

        print(f"\n[L2Demo] 开始演示 ({duration}秒)")
        print("[L2Demo] 按 Ctrl+C 停止\n")

        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                # 截图
                frame = capture.grab()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # 获取玩家输入（这里模拟无输入）
                player_action = None

                # 获取辅助动作
                action = self.runner.step(frame, player_action)

                # 执行动作
                if action.action_id > 0:
                    executor.execute(action.action_id)

                # 执行鼠标修正
                if action.mouse_dx != 0 or action.mouse_dy != 0:
                    import pydirectinput
                    dx = int(action.mouse_dx * 100)
                    dy = int(action.mouse_dy * 100)
                    pydirectinput.moveRel(dx, dy, relative=True)

                # 打印状态
                if self.runner._step_count % 30 == 0:
                    print(f"[Step {self.runner._step_count}] "
                          f"动作: {action.source} | "
                          f"置信度: {action.confidence:.2f} | "
                          f"鼠标: ({action.mouse_dx:.2f}, {action.mouse_dy:.2f})")

                # 控制帧率
                time.sleep(0.1)  # 10 FPS

        except KeyboardInterrupt:
            print("\n[L2Demo] 演示停止")

        finally:
            # 打印统计
            self.runner.print_stats()
            capture.release()
            executor.release_all()


if __name__ == "__main__":
    demo = L2AssistDemo()
    demo.run(duration=60)
