import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random

class PPO(nn.Module):
    def __init__(self, observation_width, observation_height, action_space, model_file, log_file):
        super(PPO, self).__init__()
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.action_space = action_space
        self.model_file = model_file
        self.log_file = log_file

        # 定义网络结构
        self.network = self.create_network()
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)

        # 其他PPO相关参数
        self.clip_param = 0.2
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.01

        # 初始化epsilon和相关的属性
        self.INITIAL_EPSILON = 1.0  # 初始探索率
        self.FINAL_EPSILON = 0.01  # 最终探索率
        self.epsilon = self.INITIAL_EPSILON  # 当前探索率u

        # 经验回放缓冲区
        self.replay_buffer = deque()

        self.batch_size = 16

    def create_network(self):
        # 定义一个简单的卷积神经网络
        network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (self.observation_width // 4) * (self.observation_height // 4), 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space),
            nn.Softmax(dim=-1)
        )
        return network

    def forward(self, x):
        return self.network(x)

    def save_model(self):
        torch.save(self.network.state_dict(), self.model_file)

    def load_model(self):
        self.network.load_state_dict(torch.load(self.model_file))

    def select_action(self, station):
        # # 确保station的形状是正确的，即 (HEIGHT, WIDTH, 3)
        # # 如果station是(HEIGHT, WIDTH, channels)格式，需要转换为(channels, HEIGHT, WIDTH)
        # station = station.transpose(2, 0, 1)
        #
        # # 将状态转换为torch张量，并添加批次维度
        # state = torch.FloatTensor(station).unsqueeze(0)

        # 使用策略网络获取动作概率
        action_probs = self.network(station)

        # 根据概率选择动作
        if random.random() <= self.epsilon:
            # 如果小于等于epsilon，随机选择一个动作
            action = np.random.randint(0, self.action_space)
        else:
            # 如果大于epsilon，选择概率最高的动作
            action = np.argmax(action_probs.cpu().detach().numpy())

        # 逐步减少epsilon值
        self.epsilon = max(self.FINAL_EPSILON, self.epsilon - (self.INITIAL_EPSILON - self.FINAL_EPSILON) / 10000)

        return action


    def store_transition(self, state, action, reward, next_state, done):
        # 确保 state 和 next_state 是张量
        assert isinstance(state, torch.Tensor), "State should be a tensor"
        assert isinstance(next_state, torch.Tensor), "Next state should be a tensor"
        print("Storing transition:", state.shape, action, reward, next_state.shape, done)
        # 将数据存储到经验回放缓冲区中
        self.replay_buffer.append((state, action, reward, next_state, done))

    def Train_Network(self, BATCH_SIZE, num_step):
        # 从经验回放缓冲区中采样一批数据
        batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), self.batch_size))

        print("Sampled batch:", [len(b) for b in batch])
        # 转换为张量
        states, actions, rewards, next_states, dones = zip(*batch)
        print("States shape:", states[0].shape if states else "Empty")

        # # states = torch.FloatTensor(states)
        # # 假设 states 已经是一个浮点类型的张量
        # if not states.is_floating_point():
        #     states = states.float()

        # 确保 states 是一个张量列表
        # if isinstance(states, list) and all(isinstance(s, torch.Tensor) for s in states):
        #     states = torch.stack(states)  # 将列表中的张量合并为一个批量张量
        # else:
        #     raise ValueError("States should be a list of tensors")

        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        # next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        masks = 1 - dones

        # 计算价值估计和动作概率
        values = self.network.values(states)
        action_probs = self.network.policy(states)
        next_values = self.network.values(next_states)
        dist = self.network.dist(action_probs)
        action_log_probs = dist.log_prob(actions)

        # 计算目标价值和优势函数
        targets = rewards + self.gamma * next_values * masks
        advantages = targets - values

        # 计算策略比率和比例裁剪
        ratio = torch.exp(action_log_probs - self.last_action_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 计算价值损失
        value_loss = (values - targets).pow(2).mean()

        # 计算总损失
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * dist.entropy().mean()

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新最后的动作概率
        self.last_action_log_probs = action_log_probs.detach()



if __name__ == '__main__':
    # 使用示例
    ppo_agent = PPO(observation_width=84, observation_height=84, action_space=6, model_file='ppo_model.pth', log_file='ppo_logs')