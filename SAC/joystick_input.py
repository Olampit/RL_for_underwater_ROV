# joystick_input.py

import numpy as np

class FakeJoystick:
    def __init__(self):
        self.t = 0.0
        self.dt = 0.1

    def get_target(self):
        # Simple sinusoidal depth oscillation and constant forward velocity
        vx = 0.3  # constant forward movement
        vy = 0.0
        vz = 0.0
        depth = 10.0 + np.sin(self.t) * 1.0  # oscillates between 9 and 11 meters
        yaw_rate = 0.0  # you can oscillate yaw later

        self.t += self.dt
        return {
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "depth": depth,
            "yaw_rate": yaw_rate
        }
