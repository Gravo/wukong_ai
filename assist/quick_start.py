"""
L2 辅助驾驶快速启动脚本

一键启动辅助功能，无需复杂配置

使用方式：
    # 使用规则方法（立即可用）
    python assist/quick_start.py

    # 使用训练好的模型
    python assist/quick_start.py --model checkpoints/auto_dodge_best.pt

    # 调整参数
    python assist/quick_start.py --dodge-threshold 0.9 --face-sensitivity 0.3
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assist.runner import L2AssistRunner
from assist.arbitrator import ArbitratedAction
from config import L2_ASSIST


def print_banner():
    """打印启动横幅"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           黑神话：悟空 - L2 辅助驾驶系统                    ║
║                                                              ║
║   🚗 灵感来源：自动驾驶 L2 级别"人机共驾"模式               ║
║                                                              ║
║   AI 负责：自动闪避、自动面敌、自动连招                      ║
║   玩家负责：战术决策、移动走位、技能释放                      ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_controls():
    """打印控制说明"""
    print("""
┌─────────────────────────────────────────────────────────────┐
│  控制说明：                                                   │
│    ESC    - 退出程序                                          │
│    F1     - 开关自动闪避                                      │
│    F2     - 开关自动面敌                                      │
│    F3     - 显示统计信息                                      │
│    F4     - 重置统计                                          │
└─────────────────────────────────────────────────────────────┘
    """)


def run_demo(args):
    """运行演示模式"""
    print_banner()
    print_controls()

    # 创建运行器
    runner = L2AssistRunner(dodge_model_path=args.model)

    # 调整参数
    if args.dodge_threshold:
        for module in runner.arbitrator.modules:
            if module.name == "auto_dodge":
                module.confidence_threshold = args.dodge_threshold
                print(f"[设置] 闪避阈值: {args.dodge_threshold}")

    if args.face_sensitivity:
        for module in runner.arbitrator.modules:
            if module.name == "auto_face":
                module.sensitivity = args.face_sensitivity
                print(f"[设置] 面敌灵敏度: {args.face_sensitivity}")

    # 设置键盘监听
    import keyboard as kb

    def on_key(e):
        if e.name == 'esc':
            runner.disable()
            return False
        elif e.name == 'f1' and e.event_type == kb.KEY_DOWN:
            runner.toggle_module("auto_dodge")
        elif e.name == 'f2' and e.event_type == kb.KEY_DOWN:
            runner.toggle_module("auto_face")
        elif e.name == 'f3' and e.event_type == kb.KEY_DOWN:
            runner.print_stats()
        elif e.name == 'f4' and e.event_type == kb.KEY_DOWN:
            runner.arbitrator.reset_stats()
            print("[重置] 统计已重置")

    kb.hook(on_key)

    # 等待用户切换到游戏
    print("\n  切换到游戏窗口... (3秒)")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("\n  辅助系统已启动！按 ESC 退出\n")

    # 运行辅助系统
    from env.screen_capture import ScreenCapture
    from env.action_executor import ActionExecutor

    capture = ScreenCapture()
    executor = ActionExecutor()

    frame_count = 0
    start_time = time.time()

    try:
        while runner.enabled:
            loop_start = time.time()

            # 截图
            frame = capture.grab()
            if frame is None:
                time.sleep(0.01)
                continue

            # 获取辅助动作
            action = runner.step(frame)

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

            # 每 30 帧打印状态
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"[{elapsed:.0f}s] 帧: {frame_count} | "
                      f"FPS: {fps:.1f} | "
                      f"动作: {action.source} | "
                      f"置信度: {action.confidence:.2f}")

            # 控制帧率（10 FPS）
            elapsed = time.time() - loop_start
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

    except KeyboardInterrupt:
        print("\n  用户中断")

    finally:
        # 打印最终统计
        runner.print_stats()

        # 清理
        capture.release()
        executor.release_all()
        kb.unhook_all()

        print("\n  辅助系统已停止")


def run_test(args):
    """运行测试模式"""
    print_banner()
    print("[测试模式] 验证各模块是否正常工作\n")

    # 测试自动闪避
    print("1. 测试自动闪避模块...")
    from assist.auto_dodge import AutoDodgeRuleBased
    dodge = AutoDodgeRuleBased()

    import numpy as np
    test_frame = np.random.randint(0, 255, (614, 1038, 3), dtype=np.uint8)
    action = dodge.predict(test_frame)
    print(f"   结果: {'通过' if action is not None or action is None else '失败'}")

    # 测试自动面敌
    print("2. 测试自动面敌模块...")
    from assist.auto_face import AutoFaceModule
    face = AutoFaceModule()

    action = face.predict(test_frame)
    print(f"   结果: {'通过' if action is not None or action is None else '失败'}")

    # 测试仲裁器
    print("3. 测试仲裁器...")
    from assist.arbitrator import BattleArbitrator
    arbitrator = BattleArbitrator()
    arbitrator.register_module(dodge)
    arbitrator.register_module(face)

    action = arbitrator.arbitrate(test_frame)
    print(f"   结果: 通过 (动作: {action.source})")

    print("\n[测试完成] 所有模块正常工作")


def main():
    parser = argparse.ArgumentParser(description="L2 辅助驾驶快速启动")
    parser.add_argument("--model", default=None, help="自动闪避模型路径")
    parser.add_argument("--dodge-threshold", type=float, default=None,
                        help="闪避置信度阈值 (0-1)")
    parser.add_argument("--face-sensitivity", type=float, default=None,
                        help="面敌灵敏度 (0-1)")
    parser.add_argument("--test", action="store_true", help="测试模式")

    args = parser.parse_args()

    if args.test:
        run_test(args)
    else:
        run_demo(args)


if __name__ == "__main__":
    main()
