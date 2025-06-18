#dac/dac_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DeterministicGCActor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()
        self.net = MLP(state_dim + goal_dim, action_dim)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        action = torch.tanh(self.net(x))
        return action


class DeterministicCritic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()
        self.q_net = MLP(state_dim + goal_dim + action_dim, 1)

    def forward(self, state, goal, action):
        x = torch.cat([state, goal, action], dim=-1)
        return self.q_net(x).view(-1)  # Replaces .squeeze(-1)


class PrioritizedGCReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        self.pos = 0

    def push(self, state, goal, action, reward, next_state, done):
        max_prio = max(self.priorities, default=1.0)
        data = (state, goal, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = data
            self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty.")

        prios = np.array(self.priorities)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, goals, actions, rewards, next_states, dones = map(np.stack, zip(*samples))

        return (
            torch.FloatTensor(states),
            torch.FloatTensor(goals),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1),
            torch.FloatTensor(weights).unsqueeze(1),
            indices,
        )

    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            if isinstance(p, (np.ndarray, list)):
                scalar = float(np.ravel(p)[0])
            else:
                scalar = float(p)
            self.priorities[i] = scalar + 1e-5


    def __len__(self):
        return len(self.buffer)


class DeterministicGCAgent:
    def __init__(self, state_dim, goal_dim, action_dim, device="cpu", gamma=0.99, lr=3e-4):
        self.device = device
        self.gamma = gamma

        self.actor = DeterministicGCActor(state_dim, goal_dim, action_dim).to(device)
        self.critic = DeterministicCritic(state_dim, goal_dim, action_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = PrioritizedGCReplayBuffer(capacity=100_000)

    def select_action(self, state, goal, noise_std=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        action = self.actor(state, goal).cpu().data.numpy()[0]
        action += np.random.normal(0, noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0)

    def update(self, batch_size=128, beta=0.4):
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0

        s, g, a, r, s2, d, w, idx = self.replay_buffer.sample(batch_size, beta=beta)
        s, g, a, r, s2, d, w = s.to(self.device), g.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device), w.to(self.device)

        with torch.no_grad():
            a2 = self.actor(s2, g)
            q_target = r + self.gamma * (1 - d) * self.critic(s2, g, a2).unsqueeze(1)


        q_val = self.critic(s, g, a).unsqueeze(1)
        td_error = (q_val - q_target).abs().detach().cpu().numpy()
        td_error = np.clip(td_error, 1e-6, 1e2)

        critic_loss = (F.mse_loss(q_val, q_target, reduction='none') * w).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        pred_action = self.actor(s, g)
        actor_loss = -self.critic(s, g, pred_action).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.replay_buffer.update_priorities(idx, td_error)

        return critic_loss.item(), actor_loss.item()
