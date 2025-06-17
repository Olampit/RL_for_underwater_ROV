# sac/sac_agent.py
import torch
import torch.nn.functional as F
import numpy as np
from sac.networks import Actor, Critic
from sac.replay_buffer import ReplayBuffer

class SACAgent:
    """
    Soft Actor-Critic agent implementing actor-critic updates with entropy regularization.
    """
    def __init__(self, state_dim, action_dim, device="cpu", gamma=0.99, tau=0.005, alpha=0.2, automatic_entropy_tuning=True):
        """
        Initializes the SAC agent with actor, critics, replay buffer and optimizers.

        Parameters:
            state_dim (int): Dimension of input state.
            action_dim (int): Dimension of action.
            device (str): Device to run the model on ("cpu" or "cuda").
            gamma (float): Discount factor.
            tau (float): Soft update factor for target critic.
            alpha (float): Initial entropy coefficient.
            automatic_entropy_tuning (bool): Whether to learn alpha automatically.
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer(100000)

        if self.automatic_entropy_tuning:
            self.target_entropy = -np.prod(action_dim).item()
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-2)
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.alpha = torch.tensor(alpha).to(device)
            self.log_alpha = None
            self.alpha_optimizer = None
            
        self.raw_actor = Actor(state_dim, action_dim).to(device)

        # JIT compile for fast inference
        dummy_input = torch.zeros(1, state_dim, dtype=torch.float32).to(device)
        self.actor = torch.jit.trace(self.raw_actor, dummy_input)


    def select_action(self, state, deterministic=False):
        """
        Selects an action given a state using the actor network.

        Parameters:
            state (np.ndarray): Current state.
            deterministic (bool): Whether to use mean or sample from distribution.

        Returns:
            np.ndarray: Action in [-1, 1] per dimension.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, std = self.actor(state)  # use JIT-traced model for fast inference
        if deterministic:
            action = torch.tanh(mean)
        else:
            # Use raw model for sampling (has .sample())
            action, _ = self.raw_actor.sample(state)
        return action.detach().cpu().numpy()[0]


    def update(self, batch_size=256, allow_actor_update=True):
        """
        Performs one Soft Actor-Critic update step on the critic, actor (optional), and entropy coefficient (if enabled).

        Parameters:
            batch_size (int): Number of transitions to sample from the replay buffer.
            allow_actor_update (bool): Whether to perform the actor (policy) update. Set to False during warm-up.

        Returns:
            Tuple[float, float, float]: Critic loss, actor loss, entropy (scalar values).
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0, 0.0

        # === Sample a batch ===
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        # === Critic update ===
        with torch.no_grad():
            next_action, next_log_prob = self.raw_actor.sample(next_state)
            q1_next, q2_next = self.target_critic(next_state, next_action)
            min_q_next = torch.min(q1_next, q2_next)
            entropy_term = self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * (min_q_next - entropy_term)

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # === Entropy (alpha) update ===
        action_sampled, log_prob = self.raw_actor.sample(state)

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
            self.alpha = alpha.detach()
            self.alpha = torch.clamp(self.log_alpha.exp(), min=1e-3).detach()
        else:
            alpha = self.alpha  # fixed scalar
            alpha_loss = torch.tensor(0.0)

        # === Actor update (optional) ===
        if allow_actor_update:
            q1_pi, q2_pi = self.critic(state, action_sampled)
            actor_loss = (alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
        else:
            actor_loss = torch.tensor(0.0)

        # === Soft target update ===
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        entropy = -log_prob.mean().item()
        return critic_loss.item(), actor_loss.item(), entropy


    def sample(self, state, structured=False):
        """
        Samples an action and log-prob from the policy. If `structured` is True,
        biases the sample toward physically meaningful symmetry:
        - Left (1 & 3) and right (2 & 4) pairs: same thrust
        - Vertical motors (5–8): symmetric up/down thrust

        Parameters:
            state (torch.Tensor): Input state tensor.
            structured (bool): Whether to use structured exploration.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: action, log-probability.
        """
        mean, std = self.actor(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()

        if structured:
            base = x_t.clone()

            left_thrust = torch.tanh(torch.randn_like(base[:, :1]) * 0.5)
            right_thrust = torch.tanh(torch.randn_like(base[:, :1]) * 0.5)
            base[:, 0] = left_thrust[:, 0]
            base[:, 2] = left_thrust[:, 0]
            base[:, 1] = right_thrust[:, 0]
            base[:, 3] = right_thrust[:, 0]

            yaw_bias = torch.tanh(torch.randn_like(base[:, :1]) * 0.3)
            base[:, 0] += yaw_bias[:, 0] * 0.2   # front left
            base[:, 1] -= yaw_bias[:, 0] * 0.2   # front right
            base[:, 2] += yaw_bias[:, 0] * 0.2   # rear left
            base[:, 3] -= yaw_bias[:, 0] * 0.2   # rear right

            for i in range(4, 8):
                base[:, i] = torch.tanh(torch.randn_like(base[:, i:i+1]) * 0.5).squeeze()

            x_t = base


        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)

        return action, log_prob




    @torch.no_grad()
    def get_q_value(self, state, action=None):
        """
        Estimates the Q-value of a (state, action) pair using the critic network.

        Parameters:
            state (np.ndarray): Environment state.
            action (np.ndarray or None): Action to evaluate. If None, sample from actor.

        Returns:
            float: Estimated Q-value.
        """
        if isinstance(state, dict):
            raise TypeError("get_q_value() expected an array, not a dict. Use env._state_to_obs(state) first.")
        self.critic.eval()
        self.raw_actor.eval()  # not the JIT one — it has sampling

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if action is None:
            # Sample an action from the actor
            action_tensor, _ = self.raw_actor.sample(state_tensor)
        else:
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)

        # Evaluate both critics
        q1, q2 = self.critic(state_tensor, action_tensor)
        mean_q = (q1 + q2) / 2.0

        return mean_q.cpu().item()
