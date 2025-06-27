#joystick_input.py

import numpy as np

import random

class FakeJoystick:
    def __init__(self, seed=42, total_phases=24, phase_length=25_000):
        self.episode = 0
        self.phase_length = phase_length
        self.total_phases = total_phases
        random.seed(seed)

        self.goal_schedule = self._generate_goal_schedule()
        self.goal = self._generate_goal()

    def _generate_goal_schedule(self):
        """
        Generate a list of structured, randomized goals with up to 2 nonzero components.
        """
        directions = ["vx", "vy", "vz", "yaw_rate", "pitch_rate", "roll_rate"]
        goal_list = []

        for _ in range(self.total_phases):
            goal = {d: 0.0 for d in directions}
            active = random.sample(directions, k=random.choice([1, 2]))  # choose 1 or 2 active axes

            for axis in active:
                if "v" in axis:
                    goal[axis] = random.choice([0.2, 0.4, 0.6, 0.8]) * random.choice([1, -1])
                else:  # rotational rates
                    goal[axis] = random.choice([0.1, 0.2, 0.3]) * random.choice([1, -1])

            goal_list.append(goal)

        return goal_list

    def _generate_goal(self):
        """
        Picks the goal for the current episode based on the schedule.
        """
        phase_idx = min(self.episode // self.phase_length, self.total_phases - 1)
        raw_goal = self.goal_schedule[phase_idx]

        return {k: {"mean": v, "std": 0.0} for k, v in raw_goal.items()}

    def next_step(self):
        self.episode += 1
        self.goal = self._generate_goal()

    def get_target(self):
        return self.goal

