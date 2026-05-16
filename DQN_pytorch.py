import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np


class DQN(nn.Module):
    def __init__(self, observation_width, observation_height, action_space, model_file, log_file):
        super(DQN, self).__init__()
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space
        self.replay_buffer = deque()
        self.model_file = model_file
        self.log_file = log_file
        self.epsilon = 1.0  # Initial epsilon value
        self.epsilon_decay = 0.995  # Epsilon decay rate
        self.min_epsilon = 0.01  # Minimum epsilon value
        self.gamma = 0.99  # Discount factor
        self.batch_size = 64  # Batch size for training
        self.learning_rate = 0.001  # Learning rate
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.create_Q_network()

    def create_Q_network(self):
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1)

        # Calculate the dimensions of the output from the last convolutional layer
        conv_out_size = self.calculate_conv_output(self.state_w, self.state_h)

        # Define the fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, self.action_dim)

        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def calculate_conv_output(self, width, height):
        # Calculate the output size of the convolutional layers
        width = (width - 5 + 2 * 1) // 2
        height = (height - 5 + 2 * 1) // 2
        width = (width - 5 + 2 * 1) // 2
        height = (height - 5 + 2 * 1) // 2
        return width * height * 64

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save_model(self):
        torch.save(self.state_dict(), self.model_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.model_file))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state = torch.tensor([state], dtype=torch.float).unsqueeze(0)
                q_values = self.forward(state)
                return np.argmax(q_values.cpu().numpy())

    def store_data(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > 10000:  # Replay buffer size limit
            self.replay_buffer.popleft()

    def train_network(self, BATCH_SIZE, num_step):  # self, BATCH_SIZE, num_step
        pass
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.tensor([data[0] for data in minibatch], dtype=torch.float)
        action_batch = torch.tensor([data[1] for data in minibatch], dtype=torch.float)
        reward_batch = torch.tensor([data[2] for data in minibatch], dtype=torch.float)
        next_state_batch = torch.tensor([data[3] for data in minibatch], dtype=torch.float)

        # Calculate Q values and target Q values
        current_q_values = self.forward(state_batch).gather(1, action_batch.long().unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network.forward(next_state_batch).detach().max(1)[0]
        expected_q_values = reward_batch + self.gamma * (
                    1 - torch.tensor([data[4] for data in minibatch], dtype=torch.float)) * next_q_values

        # Calculate loss
        loss = self.criterion(current_q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        pass
        # target_network.load_state_dict(self.state_dict())


if __name__ == '__main__':

    # Example usage
    dqn = DQN(observation_width=84, observation_height=84, action_space=6, model_file='dqn_model.pth', log_file='logs')