import pickle
import random
import numpy as np

class ReplayBuffer:
    """
    Replay buffer for storing transitions and sampling batches for training.
    """
    def __init__(self, capacity):
        """
        Initializes the circular buffer.

        Parameters:
            capacity (int): Maximum number of transitions stored.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Stores a transition in the buffer.

        Parameters:
            state (np.ndarray)
            action (np.ndarray)
            reward (float)
            next_state (np.ndarray)
            done (bool)
        """
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def save(self, path):
        """
        Serializes the buffer to disk using pickle.

        Parameters:
            path (str): File path to save.
        """
        with open(path, "wb") as f:
            pickle.dump({
                "capacity": self.capacity,
                "buffer": self.buffer,
                "position": self.position,
            }, f)
        print(f"[SAVE] Replay buffer saved to {path}")

    def load(self, path):
        """
        Loads a buffer from a pickle file.

        Parameters:
            path (str): File path to load.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.capacity = data["capacity"]
            self.buffer = data["buffer"]
            self.position = data["position"]
        print(f"[LOAD] Replay buffer loaded from {path}")
        
    def __len__(self):
        """
        Returns:
            int: Current number of stored transitions.
        """
        return len(self.buffer)

    def sample(self, batch_size):
        """
        Samples a batch of transitions.

        Parameters:
            batch_size (int)

        Returns:
            Tuple of np.ndarray: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
