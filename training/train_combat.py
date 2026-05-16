"""
train_combat.py - PPO战斗训练主脚本
黑神话：悟空 虎先锋战斗AI训练
"""
import os
import sys
import time
import argparse
import numpy as np

import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PPO, TRAIN, NUM_ACTIONS
from env.wukong_env import WukongEnv
from models.ppo_agent import PPOAgent
from utils.replay_buffer import RolloutBuffer
from utils.logger import TrainingLogger


def train(args):
    """主训练循环"""
    
    # 初始化环境
    env = WukongEnv(capture_backend=args.capture_backend)
    
    # 初始化智能体
    agent = PPOAgent(device=args.device)
    
    # 初始化缓冲区
    buffer = RolloutBuffer(
        rollout_length=PPO["rollout_length"],
        gamma=PPO["gamma"],
        gae_lambda=PPO["gae_lambda"],
    )
    
    # 初始化日志
    logger = TrainingLogger(
        log_dir=TRAIN["log_dir"],
        use_tensorboard=not args.no_tensorboard,
    )
    
    # 加载检查点
    if args.resume:
        agent.load(args.resume)
        print(f"[Train] 从检查点恢复: {args.resume}")
    
    # 训练变量
    total_steps = 0
    episode = 0
    best_reward = -float("inf")
    
    print(f"\n{'='*60}")
    print(f"  黑神话：悟空 - 虎先锋战斗AI训练")
    print(f"  设备: {agent.device}")
    print(f"  目标总步数: {TRAIN['total_frames']:,}")
    print(f"  Rollout长度: {PPO['rollout_length']}")
    print(f"  动作空间: {NUM_ACTIONS}")
    print(f"{'='*60}\n")
    
    try:
        while total_steps < TRAIN["total_frames"]:
            # === 收集一个Rollout ===
            buffer.reset()
            obs, info = env.reset()
            visual_obs = env.get_visual_obs()
            
            episode_reward = 0
            episode_steps = 0
            episode_start = time.time()
            
            for step_in_rollout in range(PPO["rollout_length"]):
                # 选择动作
                action, log_prob, value = agent.select_action(
                    visual_obs["frames"],
                    visual_obs["blood_info"],
                    deterministic=False,
                )
                
                # 执行动作
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_visual_obs = env.get_visual_obs()
                
                # 存入缓冲区
                buffer.add(
                    frame=visual_obs["frames"],
                    blood=visual_obs["blood_info"],
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    value=value,
                    done=terminated,
                )
                
                # 更新状态
                visual_obs = next_visual_obs
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Episode结束
                if terminated or truncated:
                    episode += 1
                    logger.log_episode(
                        episode=episode,
                        reward=episode_reward,
                        length=episode_steps,
                        boss_hp=info.get("boss_hp", 1.0),
                    )
                    
                    # 保存最佳模型
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        save_path = os.path.join(
                            TRAIN["save_dir"], "best_model.pt"
                        )
                        agent.save(save_path)
                    
                    # 重置环境
                    obs, info = env.reset()
                    visual_obs = env.get_visual_obs()
                    episode_reward = 0
                    episode_steps = 0
                
                # 定期保存
                if total_steps % TRAIN["save_interval"] == 0 and total_steps > 0:
                    save_path = os.path.join(
                        TRAIN["save_dir"],
                        f"model_step_{total_steps}.pt"
                    )
                    agent.save(save_path)
                    print(f"[Train] 已保存检查点: {save_path}")
            
            # === PPO更新 ===
            # 计算最后一个状态的V(s)用于bootstrap
            with torch.no_grad():
                frames_t = torch.FloatTensor(
                    visual_obs["frames"]
                ).unsqueeze(0).to(agent.device)
                blood_t = torch.FloatTensor(
                    visual_obs["blood_info"]
                ).unsqueeze(0).to(agent.device)
                _, _, last_value, _ = agent.network.get_action(
                    frames_t, blood_t
                )
            
            rollout_data = buffer.compute_returns_and_advantages(
                last_value=last_value.item()
            )
            
            update_info = agent.update(rollout_data)
            logger.log_update(update_info)
            
            # 打印进度
            if total_steps % TRAIN["log_interval"] == 0:
                elapsed = time.time() - episode_start
                fps = total_steps / max(elapsed, 1)
                print(
                    f"[Step {total_steps:,}] "
                    f"policy_loss={update_info['policy_loss']:.4f} | "
                    f"value_loss={update_info['value_loss']:.4f} | "
                    f"entropy={update_info['entropy']:.4f} | "
                    f"fps={fps:.1f}"
                )
    
    except KeyboardInterrupt:
        print("\n[Train] 训练被用户中断")
    
    finally:
        # 保存最终模型
        final_path = os.path.join(TRAIN["save_dir"], "final_model.pt")
        agent.save(final_path)
        
        # 打印最终统计
        stats = logger.get_stats()
        print(f"\n{'='*60}")
        print(f"  训练结束")
        print(f"  总步数: {stats['total_steps']:,}")
        print(f"  总Episode: {stats['total_episodes']}")
        print(f"  最佳奖励: {stats['best_reward']:.2f}")
        print(f"  最近10集平均: {stats['avg_reward_last10']:.2f}")
        print(f"{'='*60}")
        
        env.close()
        logger.close()


def evaluate(args):
    """评估模式：加载模型，观察AI表现"""
    env = WukongEnv(capture_backend=args.capture_backend)
    agent = PPOAgent(device=args.device)
    agent.load(args.model)
    
    print(f"\n[Eval] 评估模式，模型: {args.model}")
    print("[Eval] 按 Ctrl+C 退出\n")
    
    obs, info = env.reset()
    visual_obs = env.get_visual_obs()
    episode_reward = 0
    step = 0
    
    try:
        while True:
            action, _, _ = agent.select_action(
                visual_obs["frames"],
                visual_obs["blood_info"],
                deterministic=True,
            )
            
            obs, reward, terminated, truncated, info = env.step(action)
            visual_obs = env.get_visual_obs()
            
            episode_reward += reward
            step += 1
            
            if step % 30 == 0:
                print(
                    f"[Step {step}] action={env.executor.get_action_name(action)} | "
                    f"hp={info['player_hp']:.0%} | "
                    f"boss={info['boss_hp']:.0%} | "
                    f"reward={episode_reward:.1f}"
                )
            
            if terminated or truncated:
                print(
                    f"\n[Eval结束] 步数={step}, 奖励={episode_reward:.1f}, "
                    f"boss_hp={info['boss_hp']:.0%}\n"
                )
                obs, info = env.reset()
                visual_obs = env.get_visual_obs()
                episode_reward = 0
                step = 0
    
    except KeyboardInterrupt:
        print("\n[Eval] 评估结束")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="黑神话悟空 - 虎先锋战斗AI")
    subparsers = parser.add_subparsers(dest="command")
    
    # train 子命令
    train_parser = subparsers.add_parser("train", help="开始训练")
    train_parser.add_argument("--device", default="cuda", help="训练设备")
    train_parser.add_argument("--capture-backend", default="dxcam", help="截图后端")
    train_parser.add_argument("--resume", default=None, help="恢复训练的检查点路径")
    train_parser.add_argument("--no-tensorboard", action="store_true", help="禁用TensorBoard")
    
    # eval 子命令
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument("--model", required=True, help="模型路径")
    eval_parser.add_argument("--device", default="cuda", help="推理设备")
    eval_parser.add_argument("--capture-backend", default="dxcam", help="截图后端")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    else:
        parser.print_help()
