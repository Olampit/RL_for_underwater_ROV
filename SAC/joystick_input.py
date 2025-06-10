class FakeJoystick:
    def __init__(self):
        self.episode = 0
        self.goal = self._generate_goal()

    def _generate_goal(self):
        """Generate a target velocity based on the current episode number."""
        # Simple curriculum: every 100 episodes increases complexity
        if self.episode < 100:
            return {"vx": 0.3, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0, "pitch_rate": 0.0, "roll_rate": 0.0}
        elif self.episode < 200:
            return {"vx": 0.3, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0, "pitch_rate": 0.0, "roll_rate": 0.0}
        else:
            return {"vx": 0.3, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0, "pitch_rate": 0.0, "roll_rate": 0.0}
        #else:
        #    # Randomized 3D command (bounded)
        #    return {
        #        "vx": np.random.uniform(0.1, 0.3),
        #        "vy": np.random.uniform(-0.1, 0.1),
        #        "vz": np.random.uniform(-0.1, 0.1),
        #        "yaw_rate": np.random.uniform(-0.2, 0.2),
        #        "pitch_rate": 0.0,
        #        "roll_rate": 0.0
        #    }

    def next_episode(self):
        self.episode += 1
        self.goal = self._generate_goal()

    def get_target(self):
        return self.goal