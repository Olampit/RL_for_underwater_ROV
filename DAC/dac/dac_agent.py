#dac/dac_agent.py


"""
dac_agent.py

This module defines the core agent used for training a deterministic actor-critic 
policy to control an underwater ROV. It includes the actor and critic networks,
experience replay (with optional prioritized replay), and the main update logic.

Key Components:
- Actor: outputs deterministic continuous actions given a state sequence.
- Critic: estimates Q-values for (state, action) pairs.
- GRU or MLP critic architectures supported.
- Prioritized Experience Replay (PER): optional buffer weighting based on TD error.
- Update method: implements policy gradient and critic loss with soft target updates.

"""

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


state_dimension = 12      #! Number of features per timestep in state
sequence_dimension = 5    #! Number of timesteps (sequence length)


# -----------------------------
# Basic MLP-based Critic Network
# -----------------------------
class MLPCritic(nn.Module):
    """
    Critic model using a simple 3-layer MLP.
    Expects a combined input of state and action (per timestep).
    Only the last timestep of the input sequence is used.
    """
    def __init__(self, input_dim, action_dim, output_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),  # input: (state + action)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)  # output: scalar Q-value
        )

    def forward(self, x):
        # x shape: (B, T, state+action) → only keep the last timestep
        last = x[:, -1, :]  # (B, state+action)
        return self.net(last)  # (B, 1)





# -----------------------------
# GRU-based Feature Extractor
# -----------------------------
class GRUNetwork(nn.Module):
    """
    Generic GRU block for temporal sequence processing.
    Outputs a latent feature vector for the full input sequence.
    Used in actor or critic (but best in actor).
    """
    def __init__(self, input_dim, output_dim, hidden_dim=32, num_layers=2, batch_first=True):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # final projection
        )

    def forward(self, x):
        # x: (B, T, input_dim) → process full sequence
        _, h_n = self.gru(x)       # h_n: (num_layers, B, hidden_dim)
        h = h_n[-1]                # last layer hidden state: (B, hidden_dim)
        return self.out(h)         # output shape: (B, output_dim)






# -----------------------------
# Deterministic Actor using GRU
# -----------------------------
class DeterministicGCActor(nn.Module):
    """
    Actor network that maps sequence of states to actions.
    Uses GRU to handle temporal dependencies in observations.
    Output is squashed using tanh (for bounded control).
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        features_per_state = state_dim
        self.actor = GRUNetwork(
            input_dim=features_per_state,
            output_dim=action_dim,
        )

    def forward(self, state):
        # state: (B, T, state_dim)
        assert state.dim() == 3, "Expected shape (B, seq_len, state_dim)"
        return torch.tanh(self.actor(state))  # tanh to bound action in [-1, 1]





# -----------------------------
# Deterministic Critic using MLP
# -----------------------------
class GRUCritic(nn.Module):
    """
    GRU-based critic model that processes (state, action) sequences.
    Matches the architecture and dimensions of the actor's GRUNetwork.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        input_dim = state_dim + action_dim
        output_dim = 1  # single Q-value output

        self.gru_net = GRUNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
        )

    def forward(self, state_seq, action):
        # state_seq: (B, T, state_dim)
        # action: (B, action_dim) → need to repeat across time to concat
        B, T, _ = state_seq.shape
        action_seq = action.unsqueeze(1).repeat(1, T, 1)  # (B, T, action_dim)
        x = torch.cat([state_seq, action_seq], dim=-1)    # (B, T, state+action)
        return self.gru_net(x)                            # (B, 1)





# -----------------------------
# Buffer with priority Queue
# -----------------------------
class PrioritizedGCReplayBuffer:
    """
    Replay buffer with prioritized sampling for goal-conditioned actor-critic.
    Stores sequences of states and supports importance-sampling correction.

    Parameters:
    - capacity (int): max number of transitions to store.
    - alpha (float): prioritization exponent (0 = uniform, 1 = full prioritization).
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []        # list of (s_seq, a, r, s'_seq, done)
        self.priorities = []    # same size as buffer
        self.alpha = alpha      # controls how much prioritization matters
        self.pos = 0            # position for circular overwrite

    def push(self, state_seq, action, reward, next_state_seq, done):
        """
        Add a new transition tuple to the buffer.
        Assign max priority so it is sampled quickly.
        """
        max_prio = max(self.priorities, default=1.0)
        data = (state_seq, action, reward, next_state_seq, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = data
            self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.capacity  # wrap around

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions with importance-sampling weights.
        Returns tensors for training: (s, a, r, s', done, weight, idx).
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty.")

        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha           # priority → probability
        probs /= probs.sum()                  # normalize

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Importance-sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize for stability

        # Format into tensors
        states, actions, rewards, next_states, dones = map(np.stack, zip(*samples))

        return (
            torch.FloatTensor(states),                    # (B, T, D)
            torch.FloatTensor(actions),                   # (B, A)
            torch.FloatTensor(rewards).unsqueeze(1),      # (B, 1)
            torch.FloatTensor(next_states),               # (B, T, D)
            torch.FloatTensor(dones).unsqueeze(1),        # (B, 1)
            torch.FloatTensor(weights).unsqueeze(1),      # (B, 1)
            indices                                       # for priority update
        )

    def update_priorities(self, indices, priorities):
        """
        Update priorities after learning from a sampled batch.
        Usually called using TD errors.
        """
        for i, p in zip(indices, priorities):
            # Handle different formats (scalars, arrays, etc.)
            if isinstance(p, (np.ndarray, list)):
                scalar = float(np.ravel(p)[0])
            else:
                scalar = float(p)
            scalar = float(np.abs(scalar))
            scalar = np.clip(scalar, 1e-6, 1e3)  # clip to prevent NaNs/explosions
            self.priorities[i] = scalar

    def __len__(self):
        return len(self.buffer)


class DeterministicGCAgent:     
    
    """
    Main reinforcement learning agent implementing a deterministic actor-critic architecture
    with soft target updates and optional prioritized experience replay.

    This agent controls a robotic underwater vehicle by learning to map sequences of
    recent observations to low-level motor commands using deep networks:
    
    - Actor: GRU-based deterministic policy network that outputs an 8-dimensional motor command.
    - Critic: GRU-based or MLP-based Q-function approximator.
    - Replay Buffer: Prioritized buffer for experience sampling and TD-error-driven replay.
    
    The agent uses:
    - Policy gradient to improve the actor.
    - Temporal-Difference learning (TD error) to update the critic.
    - Polyak averaging (soft updates) to update target networks.
    - Optional TensorBoard logging for training insights. (takes a lot of space on the pc)
    """
    
    
    def __init__(self, state_dim=0, action_dim=0, device="cpu", gamma=0.99, lr=3e-4, lr_end=1e-5, tau=0.01 , use_writer=False):
        
        """
        Initializes the agent and its components.

        Args:
            state_dim (int): number of features per state (e.g., 12 for IMU readings).
            action_dim (int): number of output action dimensions (e.g., 8 for motors).
            device (str): "cpu" or "cuda".
            gamma (float): discount factor for future rewards.
            lr (float): initial learning rate.
            lr_end (float): final learning rate (for decay).
            tau (float): soft update coefficient for target networks.
            use_writer (bool): if True, enables TensorBoard logging.
        """
        
        
        print(f"[Agent Init] state_dim={state_dim}, action_dim={action_dim}")

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.use_writer = use_writer
        self.actor = DeterministicGCActor(state_dim, action_dim)
        self.actor.to(device)
        self.critic1 = GRUCritic(state_dim, action_dim).to(device)
        self.critic2 = GRUCritic(state_dim, action_dim).to(device)

        self.target_critic1 = GRUCritic(state_dim, action_dim).to(device)
        self.target_critic2 = GRUCritic(state_dim, action_dim).to(device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())


        self.target_actor = DeterministicGCActor(state_dim, action_dim)
        self.target_actor.to(device)
       

        self.target_actor.load_state_dict(self.actor.state_dict())


        # Lowering B1 results in : 
        # More reactive updates (less momentum smoothing)
        # Slightly more noise in learning
        
        # B2 is more about stoping gradients from exploding, shouldnt be an issue here. 
        
        # Should probably try betas=(0.5, 0.99) for critic at first
        # Dont touch actor too much ?
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, maximize=True)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        if self.use_writer:
            self.log_dir = os.path.join("runs", "dac_agent", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None
        
        self.current_lr = lr
        self.lr_start = lr
        self.lr_end = lr_end
        
        self.state_dim = state_dim
        self.sequence_dim = sequence_dimension


        self.replay_buffer = PrioritizedGCReplayBuffer(capacity=10_000)

        
    def load_actor(self, path):
        """Load a trained actor model from a .pth file."""
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.actor.eval()
        print(f"[LOAD] Actor model loaded from {path}")



    def soft_update(self, source, target, tau): #!
        """Performs a soft (Polyak) update from source to target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    
    def select_action(self, obs, noise_std=0.0):
        
        """
        Selects an action from the current policy for a given obs.

        Args:
            obs (np.ndarray): current obs sequence (flattened or shaped).
            noise_std (float): standard deviation of exploration noise.

        Returns:
            action (np.ndarray): clipped action vector in [-1, 1].
        """
        
        
        # Convert obs to tensor if it's not already
        if not torch.is_tensor(obs):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        else:
            obs_tensor = obs.unsqueeze(0).to(self.device)

        # Ensure correct shape: (batch, seq, obs_dim)
        obs_tensor = obs_tensor.view(1, self.sequence_dim, self.state_dim)

        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]

        # Add small noise (disabled in policy mode if noise_std=0)
        action += np.random.normal(0, noise_std, size=action.shape)

        return np.clip(action, -1.0, 1.0)





    def update(self, batch_size=128, beta=0.4, total_step = None):
        """
        Performs a single actor-critic update step using replayed experience.

        Returns a dictionary of loss values and training stats.
        """
        
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
        
        # ---- TD3 update step ----
        s, a, r, s2, d, w, idx = self.replay_buffer.sample(batch_size, beta=beta)
        s, a, r, s2, d, w = (t.to(self.device) for t in (s, a, r, s2, d, w))

        gamma = self.gamma
        tau   = self.tau

        policy_noise = 0.2
        noise_clip   = 0.5
        policy_delay = 2          # update actor every 2 critic steps
        Q_CLIP       = 20.0

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(a) * policy_noise).clamp(-noise_clip, noise_clip)
            a2 = (self.target_actor(s2) + noise).clamp(-1.0, 1.0)

            # Double target critics, clipped double Q
            q1_tgt, q2_tgt = self.target_critic1(s2, a2), self.target_critic2(s2, a2)
            q_tgt_min = torch.min(q1_tgt, q2_tgt)

            # Bootstrap target
            q_target = r + gamma * (1.0 - d) * q_tgt_min
            q_target = torch.clamp(q_target, -Q_CLIP, Q_CLIP)

        # Critic losses (two critics)
        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        
        td_error = q_target - q1 #for logging purposes

        critic1_loss = (F.mse_loss(q1, q_target, reduction='none') * w).mean()
        critic2_loss = (F.mse_loss(q2, q_target, reduction='none') * w).mean()
        critic_loss  = critic1_loss + critic2_loss

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(list(self.critic1.parameters())+list(self.critic2.parameters()), 1.0)
        self.critic1_opt.step()
        self.critic2_opt.step()

        # Delayed policy update
        if total_step % policy_delay == 0:
            # Maximize Q -> minimize negative Q
            pred_action = self.actor(s)
            actor_loss  = -self.critic1(s, pred_action).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            # Soft updates
            self.soft_update(self.critic1, self.target_critic1, tau)
            self.soft_update(self.critic2, self.target_critic2, tau)
            self.soft_update(self.actor,   self.target_actor,   tau)


        
        #! Adam uses exponential moving averages internally to adapt gradients 
        #! for optimization, but it does not perform soft updates between two 
        #! networks. In RL, soft_update() is essential to smoothly copy weights 
        #! from the main network to the target network, which Adam cannot do.
        
        
        
        
        # --- Logging weights/gradients with tensorboard
        if total_step is not None:
            if self.writer is not None:
                for name, param in self.actor.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f"actor/params/{name}", param, total_step)
                        self.writer.add_histogram(f"actor/grads/{name}", param.grad, total_step)

                for name, param in self.critic1.named_parameters():
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
                self.writer.add_scalar("q_value/mean", q1.mean().item(), total_step)
                self.writer.add_scalar("q_value/std", q1.std().item(), total_step)

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
        critic_grad_norm = sum(p.grad.data.norm(2).item() for p in self.critic1.parameters() if p.grad is not None)

        actor_weight_norm = sum(p.data.norm(2).item() for p in self.actor.parameters())
        critic_weight_norm = sum(p.data.norm(2).item() for p in self.critic1.parameters())

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
        """
        Decays learning rate linearly after warmup steps.

        Helps stabilize late training.
        """
        
        
        warmup_steps = 150_000
        lr_start = lr_start
        lr_end = lr_end
        decay_steps = 1_000_000

        if total_step < warmup_steps:
            lr = lr_start * (total_step / warmup_steps)
        else:
            decay_ratio = min((total_step - warmup_steps) / (decay_steps - warmup_steps), 1.0)
            lr = lr_start * (1 - decay_ratio) + lr_end * decay_ratio

        for param_group in self.actor_opt.param_groups:
            param_group['lr'] = lr
        for param_group in self.critic1_opt.param_groups:
            param_group['lr'] = lr

        self.current_lr = lr
        
    
    @torch.no_grad()
    def sample_random_structured(self, action_dim, batch_size=1):
        """
        Generates a random but physically plausible 4-motor command pattern
        for planar ROV motion (forward + yaw).
        """

        device = self.device
        B = batch_size

        # Initialize action tensor
        base = torch.zeros((B, action_dim)).to(device)

        def rand(minval, maxval, shape=(B, 1)):
            return torch.FloatTensor(*shape).uniform_(minval, maxval).to(device)

        # Motor signs (M1–M4), use +1 if your ESC mapping doesn't require inversion
        motor_signs = torch.tensor([+1, -1, +1, -1]).float().to(device)

        # Forward and yaw commands
        forward_cmd = rand(-1.0, 1.0)
        yaw_cmd = rand(-0.5, 0.5)

        base[:, 0] = forward_cmd[:, 0] + yaw_cmd[:, 0]  # M1
        base[:, 1] = forward_cmd[:, 0] - yaw_cmd[:, 0]  # M2
        base[:, 2] = yaw_cmd[:, 0]                      # M3
        base[:, 3] = -yaw_cmd[:, 0]                     # M4

        # Add small noise
        base += torch.randn_like(base) * 0.03

        # Apply motor sign inversion and clip
        x_t = base * motor_signs
        return torch.tanh(x_t).cpu().numpy()[0]

