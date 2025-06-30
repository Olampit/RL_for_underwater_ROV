import numpy as np
import random

class FakeJoystick:
    def __init__(self, seed=42, total_phases=1000, phase_length=25_000, evaluation_mode=False):
        self.episode = 0
        self.phase_length = phase_length
        self.total_phases = total_phases
        self.evaluation_mode = evaluation_mode
        self.manual_goal = None  # used in evaluation mode
        random.seed(seed)

        self.goal_schedule = self._generate_goal_schedule()
        self.goal = self._generate_goal()

    def _generate_goal_schedule(self):
        directions = ["vx", "vy", "vz", "yaw_rate", "pitch_rate", "roll_rate"]
        goal_list = []

        for _ in range(self.total_phases):
            goal = {d: 0.0 for d in directions}
            active = random.sample(directions, k=random.choice([1, 2]))

            for axis in active:
                if "v" in axis[:,-1]:
                    goal[axis] = random.choice([0.2, 0.4, 0.6, 0.8]) * random.choice([1, -1])
                else:
                    goal[axis] = random.choice([0.1, 0.2, 0.3]) * random.choice([1, -1])

            goal_list.append(goal)

        return goal_list

    def _generate_goal(self):
        if self.evaluation_mode and self.manual_goal is not None:
            # Wrap the manual goal in expected format
            return {k: {"mean": v, "std": 0.0} for k, v in self.manual_goal.items()}
        else:
            # Curriculum goal
            phase_idx = min(self.episode // self.phase_length, self.total_phases - 1)
            raw_goal = self.goal_schedule[phase_idx]
            return {k: {"mean": v, "std": 0.0} for k, v in raw_goal.items()}

    def set_manual_goal(self, goal_dict):
        """
        Manually set a fixed goal. Activates evaluation mode.
        Example:
            {"vx": 0.4, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0, ...}
        """
        self.manual_goal = goal_dict
        self.evaluation_mode = True
        self.goal = self._generate_goal()

    def enable_training_mode(self):
        """
        Resume curriculum goal generation.
        """
        self.evaluation_mode = False
        self.manual_goal = None
        self.goal = self._generate_goal()

    def next_step(self):
        self.episode += 1
        self.goal = self._generate_goal()

    def get_target(self):
        return self.goal
