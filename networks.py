import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#确定性策略梯度算法不需要求分布，而是直接用神经网络求abc
class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_size, hidden_size=128):
        super(ActorNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),  # Apply LayerNorm after the first layer
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),  # Reduce to hidden_size
            nn.LayerNorm(hidden_size),  # Apply LayerNorm again
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),  # Integer division
            nn.LayerNorm(hidden_size//2),
            nn.ReLU()
        )
        self.a_head = nn.Linear(hidden_size // 2, 1)  # Mean for 'a'
        self.c_head = nn.Linear(hidden_size // 2, 1)  # Mean for 'c'
        self.b_head = nn.Linear(hidden_size // 2, 1)  # Mean for 'b'
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)


    def forward(self, x):
        # Shared layers
        x = self.shared(x)
        # Actor outputs
        a = torch.tanh(self.a_head(x))  # ∈ [-1, 1]
        c = torch.tanh(self.c_head(x))  # ∈ [-1, 1]
        b = torch.tanh(self.b_head(x))  # ∈ [-1, 1]
        return a, b, c

class CriticNetwork(nn.Module):
    def __init__(self, beta, state_size, action_size=1, hidden_size=128):  #We have 3 parameters need to evalue Q value
        super(CriticNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),  # Apply LayerNorm after the first layer
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),  # Reduce to hidden_size
            nn.LayerNorm(hidden_size),  # Apply LayerNorm again
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),  # Integer division
            nn.LayerNorm(hidden_size//2),
            nn.ReLU()
        )
        self.value_head = nn.Linear(hidden_size // 2, 1)  # Value function V(s)
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        # Shared layers
        x = torch.cat([state, action], dim=-1)
        x = self.shared(x)
        # Critic output
        value = self.value_head(x)
        return value


