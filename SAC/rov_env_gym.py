# rov_env_gym.py
import numpy as np
import gym
from gym import spaces
from environment import ROVEnvironment
import time
from operator import itemgetter


class ROVEnvGymWrapper(gym.Env):
    def __init__(self, rov_env: ROVEnvironment):
        super().__init__()
        self.rov = rov_env
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

    def reset(self, connection):
        self.rov.stop_motors(connection)
        state_dict = self.rov.reset()
        return self._state_to_obs(state_dict)
    
    def stop_motors(self, connection):
        self.rov.stop_motors(connection)

    def step(self, action, state):
        self._apply_action_continuous(action)
        reward = self.rov.compute_reward(state)
        done = self.rov.is_terminal(state)
        obs = self._state_to_obs(state)
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
        keys = [
            "yaw_mean", "yaw_var",
            "pitch_mean", "pitch_var",
            "roll_mean", "roll_var",
            "vx_mean", "vy_mean", "vz_mean",
            "vx_var", "vy_var", "vz_var",
            "vel_mag_avg", "vel_mag_var"
        ]
        
        # Faster than repeated state.get(...)
        getter = itemgetter(*keys)
        values = getter({k: state.get(k, 0.0) for k in keys})  # Ensure all keys exist

        return np.array(values, dtype=np.float32)

