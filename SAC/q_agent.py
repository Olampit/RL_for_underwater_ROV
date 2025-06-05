#this is the agent that is used for Q-learning. 
import numpy as np
import random
from collections import defaultdict
from functools import partial


def default_q_values(action_size):
    return np.zeros(action_size)


class QLearningAgent:
    def __init__(self, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.q_table = defaultdict(partial(default_q_values, action_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action_size = action_size

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        q_predict = self.q_table[state][action]
        q_target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (q_target - q_predict)
        self.epsilon *= self.epsilon_decay
