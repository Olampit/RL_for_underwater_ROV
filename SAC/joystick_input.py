#joystick_input.py

import numpy as np

class FakeJoystick:
    def __init__(self):
        self.episode = 0
        self.goal = self._generate_goal()

    def _generate_goal(self):
        base = {
            "vx": {"mean": 0.3, "std": 0.02},
            "vy": {"mean": 0.0, "std": 0.01},
            "vz": {"mean": 0.0, "std": 0.01},
            "yaw_rate": {"mean": 0.0, "std": 0.01},
            "pitch_rate": {"mean": 0.0, "std": 0.01},
            "roll_rate": {"mean": 0.0, "std": 0.01}
        }

        # Add mild goal jitter (~5% noise)
        for key in base:
            jitter = np.random.normal(0, base[key]["std"])
            base[key]["mean"] += jitter

        return base

    def next_episode(self):
        self.episode += 1
        self.goal = self._generate_goal()

    def get_target(self):
        return self.goal
