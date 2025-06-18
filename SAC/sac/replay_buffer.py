import torch
import numpy as np

class PrioritizedReplayBuffer:
    """
    Experience replay buffer with prioritized sampling based on TD error.

    Stores (state, action, reward, next_state, done) transitions and samples
    batches with higher probability for transitions with larger TD error.

    Attributes:
        capacity (int): Maximum number of transitions to store.
        buffer (List): The actual transition data.
        priorities (List[float]): Per-sample priority values.
        alpha (float): Exponent for importance of priority in sampling.
        pos (int): Index of the next insertion point (circular overwrite).
    """
    def __init__(self, capacity, alpha=0.6):
        """
        Initialize the prioritized replay buffer.

        Parameters:
            capacity (int): Maximum number of transitions to store.
            alpha (float): Determines how much prioritization is used (0 = uniform sampling, 1 = full prioritization).

        Called in:
            sac_agent.py > SACAgent.__init__
        """
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer, assigning it the current max priority.

        Parameters:
            state (np.ndarray): State at time t.
            action (np.ndarray): Action taken at time t.
            reward (float): Reward received.
            next_state (np.ndarray): State at time t+1.
            done (bool): Whether the episode ended at t+1.

        Called in:
            main training loop > agent.replay_buffer.push(...)
        """
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
        """
        Sample a batch of transitions based on their priorities.

        Parameters:
            batch_size (int): Number of samples to draw.
            beta (float): Degree of importance-sampling correction (0 = no correction, 1 = full correction).

        Returns:
            Tuple[torch.Tensor]: (states, actions, rewards, next_states, dones, weights, indices)

        Called in:
            sac_agent.py > SACAgent.update()
        """
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
        """
        Update the priorities of sampled transitions after TD error is computed.

        Parameters:
            indices (List[int]): Indices of the transitions to update.
            priorities (np.ndarray or List[float]): New priority values (typically TD-errors).

        Called in:
            sac_agent.py > SACAgent.update()
        """
        
        for i, p in zip(indices, priorities):
            self.priorities[i] = float(p + 1e-5)  # epsilon to avoid zero


    def __len__(self):
        """
        Return the number of elements currently in the buffer.

        Returns:
            int: Current buffer size.

        Called in:
            sac_agent.py > SACAgent.update()
        """
        return len(self.buffer)
