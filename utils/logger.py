"""
logger.py - 训练日志记录器
支持TensorBoard + 控制台 + 文件多路输出
"""
import os
import time
import json
from datetime import datetime


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir=None, use_tensorboard=True):
        self.log_dir = log_dir or "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self._tb_writer = None
        self._log_file = None
        self._start_time = time.time()
        self._step = 0
        
        # 训练统计
        self._episode_rewards = []
        self._episode_lengths = []
        self._losses = {"policy": [], "value": [], "entropy": []}
        
        # TensorBoard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
                self._tb_writer = SummaryWriter(
                    log_dir=os.path.join(self.log_dir, run_name)
                )
                print(f"[Logger] TensorBoard已启动: {self.log_dir}/{run_name}")
            except ImportError:
                print("[Logger] tensorboard未安装，仅使用文件日志")
        
        # 日志文件
        log_path = os.path.join(
            self.log_dir,
            f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self._log_file = open(log_path, "a", encoding="utf-8")
        self._log_file.write(f"=== Training started at {datetime.now()} ===\n")
    
    def log_step(self, step, **metrics):
        """记录一步训练指标"""
        self._step = step
        
        # TensorBoard
        if self._tb_writer:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(key, value, step)
        
        # 文件日志
        if step % 1000 == 0:
            log_str = f"Step {step}: " + " | ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            )
            self._log_file.write(log_str + "\n")
            self._log_file.flush()
    
    def log_episode(self, episode, reward, length, **info):
        """记录一个episode的结果"""
        self._episode_rewards.append(reward)
        self._episode_lengths.append(length)
        
        if self._tb_writer:
            self._tb_writer.add_scalar("episode/reward", reward, episode)
            self._tb_writer.add_scalar("episode/length", length, episode)
            for k, v in info.items():
                self._tb_writer.add_scalar(f"episode/{k}", v, episode)
        
        # 打印最近N个episode的平均值
        window = min(10, len(self._episode_rewards))
        avg_reward = sum(self._episode_rewards[-window:]) / window
        avg_length = sum(self._episode_lengths[-window:]) / window
        
        elapsed = time.time() - self._start_time
        print(
            f"[Episode {episode}] "
            f"reward={reward:.1f} | "
            f"avg_reward({window})={avg_reward:.1f} | "
            f"length={length} | "
            f"time={elapsed/3600:.1f}h"
        )
        
        self._log_file.write(
            f"Episode {episode}: reward={reward:.2f}, avg_reward={avg_reward:.2f}, "
            f"length={length}, time={elapsed:.0f}s\n"
        )
        self._log_file.flush()
    
    def log_update(self, update_info):
        """记录PPO更新信息"""
        for key in ["policy_loss", "value_loss", "entropy"]:
            if key in update_info:
                self._losses[key].append(update_info[key])
        
        self.log_step(
            update_info.get("update_count", 0) * PPO_CONFIG("rollout_length"),
            **update_info,
        )
    
    def get_stats(self):
        """获取训练统计摘要"""
        stats = {
            "total_episodes": len(self._episode_rewards),
            "total_steps": self._step,
            "best_reward": max(self._episode_rewards) if self._episode_rewards else 0,
            "avg_reward_last10": (
                sum(self._episode_rewards[-10:]) / len(self._episode_rewards[-10:])
                if self._episode_rewards else 0
            ),
        }
        return stats
    
    def close(self):
        """关闭日志"""
        if self._tb_writer:
            self._tb_writer.close()
        if self._log_file:
            self._log_file.write(
                f"=== Training ended at {datetime.now()} ===\n"
            )
            self._log_file.close()


def PPO_CONFIG(key):
    """辅助函数，获取PPO配置"""
    from config import PPO
    return PPO[key]
