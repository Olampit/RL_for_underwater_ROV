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
    def __init__(self, input_dim, action_dim, output_dim=1, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # input: (state + action)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # output: scalar Q-value
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
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2, batch_first=True):
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
class DeterministicCritic(nn.Module):
    """
    Critic that uses a standard MLP to estimate Q-value.
    Takes a full sequence of states + current action.
    The action is broadcast across time and concatenated with each state.
    Only the last timestep of the sequence is used for evaluation.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        features_per_state = state_dim
        self.critic = MLPCritic(
            input_dim=features_per_state + action_dim,  # concatenated input
            action_dim=action_dim,
            output_dim=1,
        )

    def forward(self, state, action):
        batch_size = state.shape[0]
        seq_len = sequence_dimension  # global var or passed in context
        state = state.view(batch_size, seq_len, state_dimension)      # (B, T, D)
        action = action.unsqueeze(1).expand(-1, seq_len, -1)          # (B, T, A)
        x = torch.cat([state, action], dim=-1)                        # (B, T, D+A)
        return self.critic(x).view(-1)                                # (B,)




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
    
    
    def __init__(self, state_dim=0, action_dim=0, device="cpu", gamma=0.99, lr=3e-4, lr_end=1e-5, tau=0.005 , use_writer=False):
        
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
        self.critic = DeterministicCritic(state_dim, action_dim)
        self.critic.to(device)

        self.target_actor = DeterministicGCActor(state_dim, action_dim)
        self.target_actor.to(device)
        self.target_critic = DeterministicCritic(state_dim, action_dim)
        self.target_critic.to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())


        # Lowering B1 results in : 
        # More reactive updates (less momentum smoothing)
        # Slightly more noise in learning
        
        # B2 is more about stoping gradients from exploding, shouldnt be an issue here. 
        
        # Should probably try betas=(0.5, 0.99) for critic at first
        # Dont touch actor too much ?
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, maximize=True)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

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

    
    def select_action(self, state, noise_std=0.01):
        
        """
        Selects an action from the current policy for a given state.

        Args:
            state (np.ndarray): current state sequence (flattened or shaped).
            noise_std (float): standard deviation of exploration noise.

        Returns:
            action (np.ndarray): clipped action vector in [-1, 1].
        """
        
        
        # Convert state to tensor if it's not already
        if not torch.is_tensor(state):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = state.unsqueeze(0).to(self.device)

        # Ensure correct shape: (batch, seq, state_dim)
        state_tensor = state_tensor.view(1, self.sequence_dim, self.state_dim)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

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
        
        s, a, r, s2, d, w, idx = self.replay_buffer.sample(batch_size, beta=beta)
        s, a, r, s2, d, w = s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device), w.to(self.device)

        with torch.no_grad():
            a2 = self.actor(s2)
            q_target = r + self.gamma + (1-d) * self.target_critic(s2, a2).unsqueeze(1)


        q_val = self.critic(s, a).unsqueeze(1)
        
        
        td_error = (q_target - q_val).detach().cpu().numpy()
        # td_error = np.clip(td_error, 1e-6, 1e2)
        
        #!change here for the weights on monday ? 
        if total_step % 1000 == 0:
            for name, param in self.actor.named_parameters():
                if param.data.dim() == 2:  # Only log matrices
                    if self.writer is not None:
                        self.writer.add_embedding(param.data, tag=f"actor/weights/{name}", global_step=total_step)


        critic_loss = (F.mse_loss(q_val, q_target, reduction='none') * w).mean() 

        self.critic_opt.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        pred_action = self.actor(s)
        actor_loss = self.critic(s, pred_action).mean()  #No - for the ascent because we put maximize = true in adam

        self.actor_opt.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        self.soft_update(self.critic, self.target_critic, self.tau)
        self.soft_update(self.actor, self.target_actor, self.tau)

        
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
        """
        Decays learning rate linearly after warmup steps.

        Helps stabilize late training.
        """
        
        
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
        """
        Generates a random but physically plausible motor command pattern
        to be used during exploration or initialization.
        """
        
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
