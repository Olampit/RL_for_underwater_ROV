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
        self.action_dim = 8 #DONT FORGET TO CHANGE ACTION SPACE IF YOU CHANGE MOTORS
        super().__init__()
        self.rov = rov_env                                         #change shape for motors here
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.state_history = []
        obs_sample = self._state_to_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=obs_sample.shape,
            dtype=np.float32
        )
        self.history_length = 5  #! sequence dimension



    def reset(self, connection):
        start_of_action_time =  time.time()
        self.rov.stop_motors(connection)
        state_dict = self.rov.reset()
        state = self.rov.get_state(start_of_action_time)
        self.state_history = [state] * self.history_length
        obs = self._state_to_obs()
        return obs

    def stop_motors(self, connection):
        self.rov.stop_motors(connection)

    def step(self, action, no_update=False):
        
        start_of_action_time =  time.time()
        
        self._apply_action_continuous(action)
        time.sleep(0.1 / SPEED_UP)
        time.sleep(0.05) #!mavlink guaranteed update time to avoid missing values
        if no_update :
            time.sleep(0.01)
        reward = self.rov.compute_reward_2()
        done = self.rov.is_terminal()
        obs = self._state_to_obs()  
        state = self.rov.get_state(start_of_action_time)
        
        self.state_history.append(state)
        if len(self.state_history) > self.history_length:
            self.state_history.pop(0)

        return obs, reward, done, {}, state

    def _apply_action_continuous(self, action):
        for i in range(self.action_dim):
            thrust = float(np.clip(action[i], -1.0, 1.0))
            pwm = int(1500 + thrust * 400)
            self.rov.connection.mav.command_long_send(
                self.rov.connection.target_system,
                self.rov.connection.target_component,
                183, 0,
                i + 1, pwm, 0, 0, 0, 0, 0
            )

    def _state_to_obs(self):
        """
        Convert list of recent state dicts to a flat observation vector.
        Each state dict must have the same keys and order.
        """
        seq_obs = []
        for state in self.state_history:
            seq_obs.append([state[k] for k in sorted(state.keys())])
        return np.array(seq_obs, dtype=np.float32)  # (history_length, features)




