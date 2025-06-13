# rov_env_gym.py
import numpy as np
import gym
from gym import spaces
from environment import ROVEnvironment
import time
from operator import itemgetter


SPEED_UP = 5#####! ALSO DEFINED IN run_training, BEWARE

class ROVEnvGymWrapper(gym.Env):
    """
    Gym-compatible wrapper around ROVEnvironment for use with reinforcement learning agents.
    """
    def __init__(self, rov_env: ROVEnvironment):
        """
        Initialize the Gym wrapper with the underlying ROV environment.

        Parameters:
            rov_env (ROVEnvironment): The low-level environment controlling the ROV.

        Called in:
            run_training.py > make_env()
        """
        super().__init__()
        self.rov = rov_env
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

    def reset(self, connection):
        """
        Resets the ROV environment and returns the initial observation.

        Parameters:
            connection: MAVLink connection used to stop motors and reset the environment.

        Returns:
            np.ndarray: Initial observation from the state dictionary.

        Called in:
            whenever we reset the rov (every episode at the beginning)
        """
        self.rov.stop_motors(connection)
        state_dict = self.rov.reset()
        return self._state_to_obs(state_dict)
    
    def stop_motors(self, connection):
        """
        Stops all motors by sending idle PWM via the low-level environment.

        Parameters:
            connection: MAVLink connection.

        Called in:
            run_training_sac.py
        """
        self.rov.stop_motors(connection)

    def step(self, action, state):
        """
        Applies an action and returns the resulting observation, reward, done, and info.

        Parameters:
            action (np.ndarray): Continuous thrust values for 8 motors.
            state (dict): Current ROV state used for reward computation.

        Returns:
            Tuple[np.ndarray, dict, bool, dict]: Observation, reward components, done flag, empty info dict.

        Called in:
            prefill_replay.py, run_training_sac.py
        """
        self._apply_action_continuous(action)
        time.sleep(0.1/SPEED_UP)
        reward = self.rov.compute_reward(state)
        done = self.rov.is_terminal(state)
        obs = self._state_to_obs(state)
        return obs, reward, done, {}

    
    def _apply_action_continuous(self, action):
        """
        Converts continuous thrust values into PWM signals and sends them via MAVLink.

        Parameters:
            action (np.ndarray): Array of 8 values in [-1, 1].

        Called in:
            step().
        """
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
        """
        Converts a dictionary-based ROV state into a numpy observation vector.

        Parameters:
            state (dict): Dictionary with yaw/pitch/roll and velocity stats.

        Returns:
            np.ndarray: Observation array of 14 float32 values.

        Called in:
            reset(), step().
        """
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

