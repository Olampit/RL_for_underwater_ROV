# rov_env_gym.py
import numpy as np
import gym
from gym import spaces
from environment import ROVEnvironment
import time

class ROVEnvGymWrapper(gym.Env):
    def __init__(self, rov_env: ROVEnvironment):
        super().__init__()
        self.rov = rov_env
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def reset(self, connection):
        self.rov.stop_motors(connection)
        state_dict = self.rov.reset()
        return self._state_to_obs(state_dict)

    def step(self, action):
        self._apply_action_continuous(action)
        next_state_dict = self.rov.get_state()
        reward = self.rov.compute_reward(next_state_dict)
        done = self.rov.is_terminal(next_state_dict)
        obs = self._state_to_obs(next_state_dict)
        return obs, reward, done, {}

    
    def _apply_action_continuous(self, action):
        for i in range(8):
            thrust = float(np.clip(action[i], -1.0, 1.0))
            pwm = int(1500 + thrust * 400)
            self.rov.connection.mav.command_long_send(
                self.rov.connection.target_system,
                self.rov.connection.target_component,
                183, 0,  # MAV_CMD_DO_SET_SERVO
                i + 1, pwm, 0, 0, 0, 0, 0
            )

    def _state_to_obs(self, state):
        return np.array([
            state.get("pitch", 0.0),
            state.get("roll", 0.0),
            state.get("vibration", 0.0),
            state.get("vel_x", 0.0),
            state.get("vel_y", 0.0),
            state.get("vel_z", 0.0),
            state.get("pos_x", 0.0),
        ], dtype=np.float32)
