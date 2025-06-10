import pickle
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "capacity": self.capacity,
                "buffer": self.buffer,
                "position": self.position,
            }, f)
        print(f"[SAVE] Replay buffer saved to {path}")

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.capacity = data["capacity"]
            self.buffer = data["buffer"]
            self.position = data["position"]
        print(f"[LOAD] Replay buffer loaded from {path}")
        
    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
