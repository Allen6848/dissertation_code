import torch
import torch.nn.functional as F
import numpy as np
from networks import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer
import torch.optim as optim
import random
import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TD3Agent:
    def __init__(self, alpha, beta, state_size, action_size, hidden_size=128, gamma=0.99, tau=0.005, action_noise=0.1,
                 policy_noise=0.2, policy_noise_clip=0.5, delay_time=2, max_size=500,
                 batch_size=256, max_action=20, learning_rate=1e-4):
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.delay_time = delay_time
        self.update_time = 0
        self.epsilon = 0.3  #这里加入一个探索衰减，前期加大探索，防止局部最优
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.1
        self.action_counter = 0
        self.max_action = max_action
        self.learning_rate = learning_rate

        self.actor = ActorNetwork(alpha=alpha, state_size=state_size, hidden_size=hidden_size)
        self.critic1 = CriticNetwork(beta=beta, state_size=state_size, action_size=action_size, hidden_size=hidden_size)
        self.critic2 = CriticNetwork(beta=beta, state_size=state_size, action_size=action_size, hidden_size=hidden_size)

        self.target_actor = ActorNetwork(alpha=alpha, state_size=state_size, hidden_size=hidden_size)
        self.target_critic1 = CriticNetwork(beta=beta, state_size=state_size, action_size=action_size, hidden_size=hidden_size)
        self.target_critic2 = CriticNetwork(beta=beta, state_size=state_size, action_size=action_size, hidden_size=hidden_size)
        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_size, action_dim=action_size,
                                   batch_size=batch_size)
        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic1_params, target_critic1_params in zip(self.critic1.parameters(),
                                                         self.target_critic1.parameters()):
            target_critic1_params.data.copy_(tau * critic1_params + (1 - tau) * target_critic1_params)

        for critic2_params, target_critic2_params in zip(self.critic2.parameters(),
                                                         self.target_critic2.parameters()):
            target_critic2_params.data.copy_(tau * critic2_params + (1 - tau) * target_critic2_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, state, epsilon, train=True):

        if self.action_counter >= self.max_action:
            return None

        self.action_counter += 1
        a = 1
        c = 0.75
        self.actor.eval()

        if random.random() < epsilon:
            # Random actions
            #a=2
            #a = 1
            #b = random.uniform(0, 5*np.pi)
            action_b = random.uniform(0, config.max_x)
            #c = 0.75
            return (a, action_b, c)

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        _, action_b, _ = self.actor.forward(state)

        noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                             dtype=torch.float).to(device)
        action_b = torch.clamp(action_b+noise, -1, 1)
        action_b = action_b * (config.max_x / 2) + (config.max_x / 2)
        self.actor.train()
        return (a, action_b.squeeze().item(), c)

    def learn(self):
        if not self.memory.ready():
            return


        states, actions, rewards, states_, terminals = self.memory.sample_buffer()
        states_tensor = torch.tensor(states, dtype=torch.float).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.float).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states_tensor = torch.tensor(states_, dtype=torch.float).to(device)
        terminals_tensor = torch.tensor(terminals).to(device)

        with torch.no_grad():
            _, next_b_tensor,_ = self.target_actor.forward(next_states_tensor)

            action_noise = torch.randn_like(next_b_tensor) * self.policy_noise
            action_noise = torch.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)
            next_b_tensor = torch.clamp(next_b_tensor + action_noise, -1, 1)
            next_b_tensor = next_b_tensor.to(next_states_tensor.device)
            print(f"states_tensor device: {states_tensor.device}")
            print(f"next_b_tensor device: {next_b_tensor.device}")
            print(f"target_critic1 device: {next(self.target_critic1.parameters()).device}")

            q1_ = self.target_critic1.forward(next_states_tensor, next_b_tensor).view(-1)
            q2_ = self.target_critic2.forward(next_states_tensor, next_b_tensor).view(-1)
            q1_[terminals_tensor] = 0.0
            q2_[terminals_tensor] = 0.0
            critic_val = torch.min(q1_, q2_)
            target = rewards_tensor + self.gamma * critic_val
        q1 = self.critic1.forward(states_tensor, actions_tensor).view(-1)
        q2 = self.critic2.forward(states_tensor, actions_tensor).view(-1)
        print("Critic value_head device:", self.critic1.value_head.weight.device)
        print("next_states_tensor device:", next_states_tensor.device)
        critic1_loss = F.mse_loss(q1, target.detach())
        critic2_loss = F.mse_loss(q2, target.detach())
        critic_loss = critic1_loss + critic2_loss
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_time += 1
        if self.update_time % self.delay_time != 0:
            return
        new_actions_tensor = self.actor.forward(states_tensor)
        _, newb, _ = new_actions_tensor
        q1 = self.critic1.forward(states_tensor, newb)
        actor_loss = -torch.mean(q1)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
    def reset_action_counter(self):  # Add a method to reset action counter at the end of each episode
        self.action_counter = 0

