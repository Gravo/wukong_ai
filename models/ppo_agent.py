"""
ppo_agent.py - PPO智能体
Proximal Policy Optimization with clipped objective
支持：GAE优势估计、优先经验回放、多mini-batch更新
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import MODEL, PPO, NUM_ACTIONS
from models.resnet_encoder import create_encoder


class BloodEmbedding(nn.Module):
    """血量信息嵌入层"""
    def __init__(self, embed_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, embed_dim),  # player_hp, boss_hp, stamina
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.fc(x)


class PPOActorCritic(nn.Module):
    """
    PPO Actor-Critic 网络
    视觉编码器(ResNet18) + 血量嵌入 → 联合特征 → Actor头 + Critic头
    """
    
    def __init__(
        self,
        num_actions=None,
        latent_dim=None,
        blood_embed_dim=None,
        hidden_dim=None,
        encoder_type=None,
    ):
        super().__init__()
        
        self.num_actions = num_actions or NUM_ACTIONS
        self.latent_dim = latent_dim or MODEL["latent_dim"]
        self.blood_embed_dim = blood_embed_dim or MODEL["blood_embed_dim"]
        self.hidden_dim = hidden_dim or MODEL["hidden_dim"]
        
        # 视觉编码器
        self.encoder = create_encoder(encoder_type)
        
        # 血量嵌入
        self.blood_embed = BloodEmbedding(self.blood_embed_dim)
        
        # 联合特征维度
        joint_dim = self.latent_dim + self.blood_embed_dim
        
        # Actor头（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(joint_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_actions),
        )
        
        # Critic头（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(joint_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """正交初始化"""
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)
        # Actor最后一层用小增益
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0.0)
    
    def forward(self, frames, blood_info):
        """
        前向传播
        
        Args:
            frames: (B, C, H, W) 视觉输入
            blood_info: (B, 3) 血量信息
        
        Returns:
            action_logits: (B, num_actions)
            value: (B, 1)
        """
        visual_feat = self.encoder(frames)
        blood_feat = self.blood_embed(blood_info)
        
        # 拼接视觉+血量特征
        joint = torch.cat([visual_feat, blood_feat], dim=-1)
        
        action_logits = self.actor(joint)
        value = self.critic(joint)
        
        return action_logits, value
    
    def get_action(self, frames, blood_info, deterministic=False):
        """
        采样动作（用于环境交互）
        
        Returns:
            action: int
            log_prob: float
            value: float
            entropy: float
        """
        action_logits, value = self.forward(frames, blood_info)
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, value.squeeze(-1), entropy
    
    def evaluate_actions(self, frames, blood_info, actions):
        """
        评估给定动作（用于PPO更新）
        
        Returns:
            log_prob: (B,)
            entropy: (B,)
            value: (B,)
        """
        action_logits, value = self.forward(frames, blood_info)
        dist = Categorical(logits=action_logits)
        
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_prob, entropy, value.squeeze(-1)


class PPOAgent:
    """
    PPO智能体（包含训练逻辑）
    """
    
    def __init__(self, device=None, config=None):
        self.device = torch.device(device or TRAIN_DEVICE())
        self.config = config or PPO
        
        # 创建网络
        self.network = PPOActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config["lr"],
            eps=1e-5,
        )
        
        # 训练统计
        self._update_count = 0
    
    def select_action(self, frames, blood_info, deterministic=False):
        """
        选择动作
        
        Args:
            frames: np.ndarray (C, H, W)
            blood_info: np.ndarray (3,)
            deterministic: bool
        
        Returns:
            action: int
            log_prob: float
            value: float
        """
        with torch.no_grad():
            frames_t = torch.FloatTensor(frames).unsqueeze(0).to(self.device)
            blood_t = torch.FloatTensor(blood_info).unsqueeze(0).to(self.device)
            
            action, log_prob, value, entropy = self.network.get_action(
                frames_t, blood_t, deterministic=deterministic
            )
        
        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )
    
    def update(self, rollout_buffer):
        """
        PPO更新
        
        Args:
            rollout_buffer: 包含一个rollout的数据
        
        Returns:
            dict: 训练统计信息
        """
        config = self.config
        
        # 提取数据
        frames = torch.FloatTensor(rollout_buffer["frames"]).to(self.device)
        blood = torch.FloatTensor(rollout_buffer["blood_info"]).to(self.device)
        actions = torch.LongTensor(rollout_buffer["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_buffer["log_probs"]).to(self.device)
        returns = torch.FloatTensor(rollout_buffer["returns"]).to(self.device)
        advantages = torch.FloatTensor(rollout_buffer["advantages"]).to(self.device)

        # Free the original buffer data from GPU after copying to tensors
        # (rollout_buffer is a dict from compute_returns_and_advantages)
        
        # 优势归一化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        batch_size = len(actions)
        minibatch_size = min(config["minibatch_size"], batch_size)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for _ in range(config["ppo_epochs"]):
            # 随机打乱
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                mb_frames = frames[mb_indices]
                mb_blood = blood[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # 评估当前策略
                new_log_probs, entropy, values = self.network.evaluate_actions(
                    mb_frames, mb_blood, mb_actions
                )
                
                # 策略损失（PPO clip）
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - config["clip_epsilon"],
                    1.0 + config["clip_epsilon"],
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(values, mb_returns)
                
                # 熵奖励
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = (
                    policy_loss
                    + config["value_coef"] * value_loss
                    + config["entropy_coef"] * entropy_loss
                )
                
                # 梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    config["max_grad_norm"],
                )
                self.optimizer.step()
                
                # Free minibatch tensors
                del mb_frames, mb_blood, mb_actions, mb_old_log_probs
                del mb_returns, mb_advantages
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        self._update_count += 1
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "update_count": self._update_count,
        }
    
    def save(self, path):
        """保存模型"""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self._update_count,
        }, path)
        print(f"[PPOAgent] 模型已保存到 {path}")
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._update_count = checkpoint.get("update_count", 0)
        print(f"[PPOAgent] 模型已从 {path} 加载，update_count={self._update_count}")


def TRAIN_DEVICE():
    """获取训练设备"""
    return "cuda" if torch.cuda.is_available() else "cpu"
