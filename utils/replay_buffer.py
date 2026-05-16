"""
replay_buffer.py - Rollout缓冲区 + 优先经验回放
PPO使用on-policy的rollout buffer，同时提供PER用于off-policy扩展
"""
import numpy as np
from config import PPO, REPLAY


class RolloutBuffer:
    """
    On-policy Rollout缓冲区
    收集一个rollout的数据，用于PPO更新后清空
    """
    
    def __init__(self, rollout_length=None, gamma=None, gae_lambda=None):
        self.rollout_length = rollout_length or PPO["rollout_length"]
        self.gamma = gamma or PPO["gamma"]
        self.gae_lambda = gae_lambda or PPO["gae_lambda"]
        
        self.reset()
    
    def reset(self):
        """清空缓冲区"""
        self.frames = []
        self.blood_info = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, frame, blood, action, log_prob, reward, value, done):
        """
        添加一步数据
        
        Args:
            frame: np.ndarray (C, H, W)
            blood: np.ndarray (3,)
            action: int
            log_prob: float
            reward: float
            value: float
            done: bool
        """
        self.frames.append(frame)
        self.blood_info.append(blood)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def __len__(self):
        return len(self.rewards)
    
    def is_full(self):
        return len(self) >= self.rollout_length
    
    def compute_returns_and_advantages(self, last_value=0.0):
        """
        计算GAE优势和折扣回报
        
        Args:
            last_value: 最后一个状态的V(s)估计值（用于bootstrap）
        
        Returns:
            dict: 包含所有数据的字典，包括计算好的returns和advantages
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        
        # 从后往前计算GAE
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            # TD误差
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            
            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        # Returns = Advantages + Values
        returns = advantages + np.array(self.values, dtype=np.float32)
        
        return {
            "frames": np.array(self.frames, dtype=np.float32),
            "blood_info": np.array(self.blood_info, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.int64),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "returns": returns,
            "advantages": advantages,
        }


class PrioritizedReplayBuffer:
    """
    优先经验回放（PER）缓冲区
    用于off-policy学习或经验复用
    """
    
    def __init__(
        self,
        buffer_size=None,
        alpha=None,
        beta_start=None,
        beta_frames=None,
    ):
        self.buffer_size = buffer_size or REPLAY["buffer_size"]
        self.alpha = alpha or REPLAY["alpha"]
        self.beta_start = beta_start or REPLAY["beta_start"]
        self.beta_frames = beta_frames or REPLAY["beta_frames"]
        
        self._buffer = []
        self._pos = 0
        self._priorities = np.zeros(self.buffer_size, dtype=np.float32)
        self._max_priority = 1.0
        self._frame_count = 0
    
    def add(self, experience, priority=None):
        """
        添加一条经验
        
        Args:
            experience: tuple (frame, blood, action, reward, next_frame, next_blood, done)
            priority: float，优先级（None则使用最大优先级）
        """
        if priority is None:
            priority = self._max_priority
        
        if len(self._buffer) < self.buffer_size:
            self._buffer.append(experience)
        else:
            self._buffer[self._pos] = experience
        
        self._priorities[self._pos] = priority ** self.alpha
        self._pos = (self._pos + 1) % self.buffer_size
        self._max_priority = max(self._max_priority, priority)
        self._frame_count += 1
    
    def sample(self, batch_size):
        """
        优先采样
        
        Returns:
            batch: list of experiences
            indices: np.ndarray
            weights: np.ndarray (重要性采样权重)
        """
        buffer_len = len(self._buffer)
        if buffer_len < batch_size:
            return None, None, None
        
        # 计算采样概率
        priorities = self._priorities[:buffer_len]
        probs = priorities / priorities.sum()
        
        # 采样
        indices = np.random.choice(buffer_len, batch_size, p=probs, replace=False)
        
        # 重要性采样权重
        beta = min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * (self._frame_count / self.beta_frames)
        )
        weights = (buffer_len * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # 归一化
        
        batch = [self._buffer[i] for i in indices]
        
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        """更新采样到的经验的优先级"""
        for idx, priority in zip(indices, priorities):
            self._priorities[idx] = priority ** self.alpha
            self._max_priority = max(self._max_priority, priority)
    
    def __len__(self):
        return len(self._buffer)
