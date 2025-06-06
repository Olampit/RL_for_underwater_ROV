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
        yaw_rate = 0.0  
        pitch_rate = 0.0
        roll_rate = 0.0

        self.t += self.dt
        return {
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "yaw_rate": yaw_rate,
            "pitch_rate": pitch_rate,
            "roll_rate": roll_rate

        }