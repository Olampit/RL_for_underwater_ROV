# sac/sac_agent.py
import torch
import torch.nn.functional as F
import numpy as np
from sac.networks import Actor, Critic
from sac.replay_buffer import ReplayBuffer

class SACAgent:
    def __init__(self, state_dim, action_dim, device="cpu", gamma=0.99, tau=0.005, alpha=0.2, automatic_entropy_tuning=True):
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
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, std = self.actor(state)  # use JIT-traced model for fast inference
        if deterministic:
            action = torch.tanh(mean)
        else:
            # Use raw model for sampling (has .sample())
            action, _ = self.raw_actor.sample(state)
        return action.detach().cpu().numpy()[0]


    def update(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            # Ensure consistent return type
            return 0.0, 0.0, 0.0, self.alpha.item()

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_prob = self.raw_actor.sample(next_state)
            q1_next, q2_next = self.target_critic(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * q_next

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        action_sampled, log_prob = self.raw_actor.sample(state)
        q1_pi, q2_pi = self.critic(state, action_sampled)
        actor_loss = (self.alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()
        else:
            alpha_loss = torch.tensor(0.0)

        self.actor_opt.step()

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        entropy = -log_prob.mean().item()
        return critic_loss.item(), actor_loss.item(), entropy

    def sample(self, state):
        mean, std = self(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_prob
