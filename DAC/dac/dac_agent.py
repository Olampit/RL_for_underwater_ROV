#dac/dac_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(64, 64)):
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
            #self.priorities[i] = scalar + 1e-5
            self.priorities[i] = 1e-4


    def __len__(self):
        return len(self.buffer)


class DeterministicGCAgent:
    def __init__(self, state_dim, goal_dim, action_dim, device="cpu", gamma=0.99, lr=5e-2, lr_end = 1e-4):
        self.device = device
        self.gamma = gamma

        self.actor = DeterministicGCActor(state_dim, goal_dim, action_dim).to(device)
        self.critic = DeterministicCritic(state_dim, goal_dim, action_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = PrioritizedGCReplayBuffer(capacity=10_000)
        
        dummy_state = torch.zeros(1, state_dim).to(self.device)
        dummy_goal = torch.zeros(1, goal_dim).to(self.device)
        dummy_action = torch.zeros(1, action_dim).to(self.device)

        self.writer = SummaryWriter(log_dir="runs/dac_agent")
        
        dummy_state = torch.randn(1, state_dim).to(self.device)
        dummy_goal = torch.randn(1, goal_dim).to(self.device)
        dummy_action = torch.randn(1, action_dim).to(self.device)

        self.writer.add_graph(self.actor, (dummy_state, dummy_goal))
        self.writer.add_graph(self.critic, (dummy_state, dummy_goal, dummy_action))

    
    def select_action(self, state, goal, noise_std=0.01):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        action = self.actor(state, goal).cpu().data.numpy()[0]
        action += np.random.normal(0, noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0)

    def update(self, batch_size=128, beta=0.4, total_step = None):
        if len(self.replay_buffer) < batch_size:
            return {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "td_mean": 0.0,
            "td_max": 0.0,
            "td_min": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "actor_weight_norm": 0.0,
            "critic_weight_norm": 0.0,
            "learning_rate": 0.0
        }
        
        s, g, a, r, s2, d, w, idx = self.replay_buffer.sample(batch_size, beta=beta)
        s, g, a, r, s2, d, w = s.to(self.device), g.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device), w.to(self.device)

        with torch.no_grad():
            a2 = self.actor(s2, g)
            q_target = r + self.gamma * self.critic(s2, g, a2).unsqueeze(1)


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
        
        
        
        
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"actor/params/{name}", param, total_step)
                self.writer.add_histogram(f"actor/grads/{name}", param.grad, total_step)

        for name, param in self.critic.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"critic/params/{name}", param, total_step)
                self.writer.add_histogram(f"critic/grads/{name}", param.grad, total_step)

        self.writer.add_scalar("lr/actor", self.current_lr, total_step)
        self.writer.add_scalar("lr/critic", self.current_lr, total_step)

        self.replay_buffer.update_priorities(idx, td_error)

        # Calculate norms for debug
        actor_grad_norm = sum(p.grad.data.norm(2).item() for p in self.actor.parameters() if p.grad is not None)
        critic_grad_norm = sum(p.grad.data.norm(2).item() for p in self.critic.parameters() if p.grad is not None)

        actor_weight_norm = sum(p.data.norm(2).item() for p in self.actor.parameters())
        critic_weight_norm = sum(p.data.norm(2).item() for p in self.critic.parameters())

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "td_mean": float(td_error.mean()),
            "td_max": float(td_error.max()),
            "td_min": float(td_error.min()),
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "actor_weight_norm": actor_weight_norm,
            "critic_weight_norm": critic_weight_norm,
            "learning_rate" : self.current_lr
        }



    def lr_step(self, total_step, lr_start=3e-4, lr_end=1e-6):
        warmup_steps = 5000
        lr_start = lr_start
        lr_end = lr_end
        decay_steps = 100_000

        if total_step < warmup_steps:
            lr = lr_start * (total_step / warmup_steps)
        else:
            decay_ratio = min((total_step - warmup_steps) / (decay_steps - warmup_steps), 1.0)
            lr = lr_start * (1 - decay_ratio) + lr_end * decay_ratio

        for param_group in self.actor_opt.param_groups:
            param_group['lr'] = lr
        for param_group in self.critic_opt.param_groups:
            param_group['lr'] = lr

        self.current_lr = lr