# sac/networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class MLP(nn.Module):
    """
    Simple feedforward neural network with configurable hidden layers and ReLU activations.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 128)):
        """
        Builds a multilayer perceptron with given input/output dimensions.

        Parameters:
            input_dim (int): Input size.
            output_dim (int): Output size.
            hidden_dims (tuple): Hidden layer sizes.
        """
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

class Actor(nn.Module):
    """
    Actor network that outputs a mean and std (log space) for each action dimension.
    """
    def __init__(self, state_dim, action_dim):
        """
        Builds the actor model with stochastic output.

        Parameters:
            state_dim (int): Input state size.
            action_dim (int): Output action size.
        """
        super().__init__()
        self.net = MLP(state_dim, action_dim * 2)  # Outputs mean and log_std

    def forward(self, state):
        """
        Computes the mean and std of the action distribution.

        Parameters:
            state (torch.Tensor): Input state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation.
        """
        mean_logstd = self.net(state)
        mean, log_std = torch.chunk(mean_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        """
        Samples an action from the policy using the reparameterization trick.

        Parameters:
            state (torch.Tensor): Input state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tanh-squashed action, log-probability.
        """
        mean, std = self(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    """
    Critic network computing two Q-value estimates for given state-action pairs.
    """
    def __init__(self, state_dim, action_dim):
        """
        Initializes the critic with two independent Q-networks.

        Parameters:
            state_dim (int): State input size.
            action_dim (int): Action input size.
        """
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1)
        self.q2 = MLP(state_dim + action_dim, 1)

    def forward(self, state, action):
        """
        Computes Q-values for the given (state, action) pair.

        Parameters:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Q1 and Q2 outputs.
        """
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)
