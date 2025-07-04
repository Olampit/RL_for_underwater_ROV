#dac/dac_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR

import os
import datetime

# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dims=(128,128,128,128)):
#         super().__init__()
#         layers = []
#         dims = [input_dim] + list(hidden_dims)

#         for i in range(len(dims) - 1):
#             layers.append(nn.Linear(dims[i], dims[i+1]))
#             # layers.append(nn.LayerNorm(dims[i+1]))  # Normalize before activation
#             layers.append(nn.ReLU())

#         layers.append(nn.Linear(dims[-1], output_dim))
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)


state_dimension = 12
sequence_dimension = 2

class GRUNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        _, h_n = self.gru(x)        # h_n: (num_layers, batch, hidden_dim)
        h = h_n[-1]                 # (batch, hidden_dim)
        return self.out(h)





class DeterministicGCActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        features_per_state = state_dim
        self.actor = GRUNetwork(
            input_dim=features_per_state,
            output_dim=action_dim,
            hidden_dim=256
        )

    def forward(self, state):
        if state.dim() == 2:
            batch_size = state.shape[0]
            seq_len = sequence_dimension
            state = state.view(batch_size, seq_len, state_dimension)
        return torch.tanh(self.actor(state))




class DeterministicCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        features_per_state = state_dim       
        self.critic = GRUNetwork(
            input_dim=features_per_state + action_dim,
            output_dim=1,
            hidden_dim=256
        )

    def forward(self, state, action):
        batch_size = state.shape[0]
        seq_len = sequence_dimension
        state = state.view(batch_size, seq_len, state_dimension)  # (B, T, 15)
        action = action.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, 8)
        x = torch.cat([state, action], dim=-1)  # (B, T, 23)
        return self.critic(x).view(-1)




class PrioritizedGCReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities, default=1.0)
        data = (state, action, reward, next_state, done)

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

        states, actions, rewards, next_states, dones = map(np.stack, zip(*samples))

        return (
            torch.FloatTensor(states),
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
    def __init__(self, state_dim=0, action_dim=0, device="cpu", gamma=0.99, lr=3e-4, lr_end=1e-5, tau=0.005, use_writer=False):
        
        print(f"[Agent Init] state_dim={state_dim}, action_dim={action_dim}")

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.use_writer = use_writer
        self.actor = DeterministicGCActor(state_dim, action_dim).to(device)
        self.critic = DeterministicCritic(state_dim, action_dim).to(device)

        self.target_actor = DeterministicGCActor(state_dim, action_dim).to(device)
        self.target_critic = DeterministicCritic(state_dim, action_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        if self.use_writer:
            self.log_dir = os.path.join("runs", "dac_agent", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None
        
        self.current_lr = lr
        self.lr_start = lr
        self.lr_end = lr_end

        self.replay_buffer = PrioritizedGCReplayBuffer(capacity=10_000)

        
        


    def soft_update(self, source, target, tau):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    
    def select_action(self, state, noise_std=0.01):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, 4, 15)
        state_tensor = state_tensor.view(1, sequence_dimension, state_dimension)
        action = self.actor(state_tensor).cpu().data.numpy()[0]
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
        
        s, a, r, s2, d, w, idx = self.replay_buffer.sample(batch_size, beta=beta)
        s, a, r, s2, d, w = s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device), w.to(self.device)

        with torch.no_grad():
            a2 = self.actor(s2)
            q_target = r + self.gamma * self.target_critic(s2, a2).unsqueeze(1)


        q_val = self.critic(s, a).unsqueeze(1)
        
        
        td_error = (q_val - q_target).abs().detach().cpu().numpy()
        td_error = np.clip(td_error, 1e-6, 1e2)
        
        #!change here for the weights on monday ? 
        if total_step % 1000 == 0:
            for name, param in self.actor.named_parameters():
                if param.data.dim() == 2:  # Only log matrices
                    if self.writer is not None:
                        self.writer.add_embedding(param.data, tag=f"actor/weights/{name}", global_step=total_step)


        critic_loss = (F.mse_loss(q_val, q_target, reduction='none') * w).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        pred_action = self.actor(s)
        actor_loss = -self.critic(s, pred_action).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        self.soft_update(self.critic, self.target_critic, self.tau)
        self.soft_update(self.actor, self.target_actor, self.tau)

        
        
        
        
        # --- Logging weights/gradients with tensorboard
        if total_step is not None:
            if self.writer is not None:
                for name, param in self.actor.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f"actor/params/{name}", param, total_step)
                        self.writer.add_histogram(f"actor/grads/{name}", param.grad, total_step)

                for name, param in self.critic.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f"critic/params/{name}", param, total_step)
                        self.writer.add_histogram(f"critic/grads/{name}", param.grad, total_step)

                # Scalar logging
                self.writer.add_scalar("loss/critic", critic_loss.item(), total_step)
                self.writer.add_scalar("loss/actor", actor_loss.item(), total_step)
                self.writer.add_scalar("td_error/mean", float(td_error.mean()), total_step)
                self.writer.add_scalar("td_error/max", float(td_error.max()), total_step)
                self.writer.add_scalar("td_error/min", float(td_error.min()), total_step)
                self.writer.add_scalar("lr/actor", self.current_lr, total_step)
                self.writer.add_scalar("lr/critic", self.current_lr, total_step)
                self.writer.add_scalar("q_value/mean", q_val.mean().item(), total_step)
                self.writer.add_scalar("q_value/std", q_val.std().item(), total_step)

                # Track action stats
                action_tensor = self.actor(s)
                self.writer.add_scalar("action/mean", action_tensor.mean().item(), total_step)
                self.writer.add_scalar("action/std", action_tensor.std().item(), total_step)

                # Optional: log one reward component if passed via r
                if r.numel() == 1:
                    self.writer.add_scalar("env/reward_total", r.item(), total_step)

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
        
    
    @torch.no_grad()
    def sample_random_structured(self, batch_size=1):
        device = self.device
        B = batch_size

        # Create base
        base = torch.zeros((B, 8)).to(device)

        def rand(minval, maxval, shape=(B, 1)):
            return torch.FloatTensor(*shape).uniform_(minval, maxval).to(device)

        motor_signs = torch.tensor([+1, -1, +1, -1, +1, +1, +1, +1]).float().to(device)

        # --- FORWARD + YAW ---
        forward_cmd = rand(-1.0, 1.0)
        yaw_cmd = rand(-0.5, 0.5)

        base[:, 0] = forward_cmd[:, 0] + yaw_cmd[:, 0]  # M1
        base[:, 1] = forward_cmd[:, 0] - yaw_cmd[:, 0]  # M2
        base[:, 2] = yaw_cmd[:, 0]                      # M3
        base[:, 3] = -yaw_cmd[:, 0]                     # M4

        # --- LIFT + PITCH + ROLL ---
        lift_cmd = rand(-0.8, 0.8)
        pitch_cmd = rand(-0.4, 0.4)
        roll_cmd = rand(-0.4, 0.4)

        base[:, 4] = lift_cmd[:, 0] + pitch_cmd[:, 0] + roll_cmd[:, 0]  # M5
        base[:, 5] = lift_cmd[:, 0] + pitch_cmd[:, 0] - roll_cmd[:, 0]  # M6
        base[:, 6] = lift_cmd[:, 0] - pitch_cmd[:, 0] + roll_cmd[:, 0]  # M7
        base[:, 7] = lift_cmd[:, 0] - pitch_cmd[:, 0] - roll_cmd[:, 0]  # M8

        # Add structured Gaussian noise
        base += torch.randn_like(base) * 0.03

        x_t = base * motor_signs
        return torch.tanh(x_t).cpu().numpy()[0]
