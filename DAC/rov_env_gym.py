# rov_env_gym.py

import numpy as np
import gym
from gym import spaces
from environment import ROVEnvironment
import time
from operator import itemgetter

SPEED_UP = 5

class ROVEnvGymWrapper(gym.Env):
    def __init__(self, rov_env: ROVEnvironment):
        super().__init__()
        self.rov = rov_env
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        obs_sample = self._state_to_obs(self.rov.get_state())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=obs_sample.shape,
            dtype=np.float32
        )
        self.history_length = 4  # or 5–10
        self.state_history = []



    def reset(self, connection):
        self.rov.stop_motors(connection)
        state_dict = self.rov.reset()
        
        state = self.env.get_state()
        self.state_history = [state] * self.history_length
        obs = self.state_to_obs()
        return obs

    def stop_motors(self, connection):
        self.rov.stop_motors(connection)

    def step(self, action, state, no_update=False):
        self._apply_action_continuous(action)
        time.sleep(0.1 / SPEED_UP)
        if no_update :
            time.sleep(0.01)
        reward = self.rov.compute_reward(state)
        done = self.rov.is_terminal(state)
        obs = self._state_to_obs(self.rov.get_state())
        return obs, reward, done, {}

    def _apply_action_continuous(self, action):
        for i in range(8):
            thrust = float(np.clip(action[i], -1.0, 1.0))
            pwm = int(1500 + thrust * 400)
            self.rov.connection.mav.command_long_send(
                self.rov.connection.target_system,
                self.rov.connection.target_component,
                183, 0,
                i + 1, pwm, 0, 0, 0, 0, 0
            )

    def state_to_obs(self):
        """
        Convert list of recent state dicts to a flat observation vector.
        Each state dict must have the same keys and order.
        """
        flat_obs = []
        for state in self.state_history:
            for key in sorted(state.keys()):
                flat_obs.append(state[key])
        return np.array(flat_obs, dtype=np.float32)


