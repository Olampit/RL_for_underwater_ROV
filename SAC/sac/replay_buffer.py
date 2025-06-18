import torch
import numpy as np

class PrioritizedReplayBuffer:
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
            self.priorities[i] = float(p + 1e-5)  # epsilon to avoid zero


    def __len__(self):
        return len(self.buffer)
    