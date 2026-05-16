"""
quick_test.py - 快速系统验证脚本
不需要游戏运行，验证所有组件可正常工作

Usage: python tools/quick_test.py
"""
import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        failed += 1


def main():
    print("=" * 60)
    print("  wukong_ai 系统快速验证")
    print("=" * 60)
    
    # --- 基础环境 ---
    print("\n[1/6] 基础环境")
    test("PyTorch + CUDA", lambda: (
        print(f"    PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, "
              f"GPU={torch.cuda.get_device_name(0)}") if torch.cuda.is_available() else None,
    ))
    test("GPU Tensor", lambda: torch.randn(10, 10).cuda().sum())
    
    # --- 配置 ---
    print("\n[2/6] 配置")
    import config
    test("config.py 加载", lambda: None)
    test(f"动作空间: {config.NUM_ACTIONS}个", lambda: assert_len(config.ACTION_SPACE, config.NUM_ACTIONS))
    test(f"编码器: {config.MODEL['encoder']}", lambda: None)
    
    # --- 截图模块 ---
    print("\n[3/6] 截图模块")
    test("dxcam import", lambda: __import__('dxcam'))
    test("mss import", lambda: __import__('mss'))
    
    # --- 血量检测 ---
    print("\n[4/6] 血量检测")
    from env.blood_detector import BloodDetector
    detector = BloodDetector()
    
    # 创建合成画面（1920x1080）
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # 在player_hp区域画红色
    r = config.BLOOD_REGION["player_hp"]
    frame[r["top"]:r["top"]+r["height"], r["left"]:r["left"]+r["width"]] = [40, 40, 200]
    
    test("BloodDetector 合成画面检测", lambda: detector.detect_hp(frame, 'player_hp'))
    
    # 测试亮度回退
    frame2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame2[40:58, 500:1420] = [200, 200, 200]  # boss区域亮色
    test("BloodDetector get_all_vitals", lambda: detector.get_all_vitals(frame2))
    
    # --- 模型 ---
    print("\n[5/6] 模型 (CNN)")
    config.MODEL['encoder'] = 'cnn'
    config.MODEL['pretrained'] = False
    
    from models.ppo_agent import PPOAgent
    agent = PPOAgent()
    
    test("PPOAgent 创建", lambda: None)
    
    frames = np.random.randn(12, 224, 224).astype(np.float32)
    blood = np.array([0.8, 0.5, 0.9], dtype=np.float32)
    
    action, log_prob, value = agent.select_action(frames, blood)
    test(f"select_action: action={action}", lambda: None)
    
    # --- 训练循环 ---
    print("\n[6/6] 训练循环")
    from utils_new.replay_buffer import RolloutBuffer
    buf = RolloutBuffer(rollout_length=32)
    
    t0 = time.time()
    for step in range(32):
        frames = np.random.randn(12, 224, 224).astype(np.float32)
        blood = np.clip(np.array([0.8 - step*0.02, 1.0 - step*0.01, 0.9], dtype=np.float32), 0, 1)
        action, log_prob, value = agent.select_action(frames, blood)
        reward = float(np.random.randn())
        buf.add(frame=frames, blood=blood, action=action, log_prob=log_prob,
                reward=reward, value=value, done=(step == 31))
    
    rollout_data = buf.compute_returns_and_advantages(last_value=0.0)
    info = agent.update(rollout_data)
    elapsed = time.time() - t0
    
    fps = 32 / elapsed
    test(f"完整训练循环 ({fps:.1f} fps)", lambda: None)
    print(f"    policy_loss={info['policy_loss']:.4f}, value_loss={info['value_loss']:.4f}, entropy={info['entropy']:.4f}")
    
    # --- ResNet18 (if weights available) ---
    if os.path.exists(os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")):
        print("\n[BONUS] ResNet18 预训练权重")
        config.MODEL['encoder'] = 'resnet18'
        config.MODEL['pretrained'] = True
        try:
            agent_resnet = PPOAgent()
            action, _, _ = agent_resnet.select_action(frames, blood)
            test("ResNet18 推理", lambda: None)
        except Exception as e:
            test("ResNet18 推理", lambda: (_ for _ in ()).throw(e))
    
    # --- 结果 ---
    print("\n" + "=" * 60)
    if failed == 0:
        print(f"  ✅ 全部通过！{passed}/{passed+failed} 项测试成功")
        print("  系统就绪，可以开始训练！")
    else:
        print(f"  ⚠️ {passed} 通过, {failed} 失败")
        print("  请检查失败项")
    print("=" * 60)
    
    return failed == 0


def assert_len(obj, expected):
    assert len(obj) == expected, f"Expected len={expected}, got {len(obj)}"


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
