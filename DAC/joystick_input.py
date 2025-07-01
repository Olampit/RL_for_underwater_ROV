import numpy as np
import random

class FakeJoystick:
    def __init__(self, seed=42, total_phases=1000, evaluation_mode=False):
        self.episode = 0
        self.total_phases = total_phases
        self.evaluation_mode = evaluation_mode
        self.manual_goal = None  # used in evaluation mode
        random.seed(seed)

        self.goal_schedule = self._generate_goal_schedule()
        self.goal_index = 0
        self.goal = self._generate_goal()

        # Success tracking
        self.success_counter = 0
        self.success_threshold = 20  # number of good steps before switching
        self.error_threshold_v = 0.1
        self.error_threshold_r = 0.1

    def _generate_goal_schedule(self):
        # Directions without pitch_rate and roll_rate
        directions = ["vx", "vy", "vz", "yaw_rate"]
        goal_list = []

        for _ in range(self.total_phases):
            # Initialize goal with pitch_rate and roll_rate set to 0
            goal = {d: 0.0 for d in directions}
            goal["pitch_rate"] = 0.0  # Always set pitch_rate to 0
            goal["roll_rate"] = 0.0   # Always set roll_rate to 0
            
            # Randomly select active axis (one or two axes)
            active = random.sample(directions, k=random.choice([1, 1]))  # Change second 1 to 2 for 2-axes

            for axis in active:
                goal[axis] = random.choice([0.1, 0.2, 0.3, 0.4,]) * random.choice([1, -1])

            goal_list.append(goal)

        # Optional: if you want to ensure that there's always a "final" goal with all zero values, add it here
        goal_list.append({"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0, "pitch_rate": 0.0, "roll_rate": 0.0})

        return goal_list

    def _generate_goal(self):
        if self.evaluation_mode and self.manual_goal is not None:
            return {k: {"mean": v, "std": 0.0} for k, v in self.manual_goal.items()}
        else:
            raw_goal = self.goal_schedule[self.goal_index]
            return {k: {"mean": v, "std": 0.0} for k, v in raw_goal.items()}

    def set_manual_goal(self, goal_dict):
        self.manual_goal = goal_dict
        self.evaluation_mode = True
        self.goal = self._generate_goal()

    def enable_training_mode(self):
        self.evaluation_mode = False
        self.manual_goal = None
        self.goal = self._generate_goal()

    def get_target(self):
        return self.goal

    def update_success_tracking(self, reward_components):
        if self.evaluation_mode:
            return  # No switching in evaluation mode

        # Get errors from reward_components
        vx_e = abs(reward_components.get("vx_error", 1.0))
        vy_e = abs(reward_components.get("vy_error", 1.0))
        vz_e = abs(reward_components.get("vz_error", 1.0))
        yaw_e = abs(reward_components.get("yaw_error", 1.0))
        pitch_e = abs(reward_components.get("pitch_error", 1.0))
        roll_e = abs(reward_components.get("roll_error", 1.0))

        # Check if all errors are within thresholds
        if (
            vx_e < self.error_threshold_v and
            vy_e < self.error_threshold_v and
            vz_e < self.error_threshold_v and
            yaw_e < self.error_threshold_r and
            pitch_e < self.error_threshold_r and
            roll_e < self.error_threshold_r
        ):
            self.success_counter += 1
        else:
            self.success_counter = 0  # reset on any error

        # Switch goal if threshold met
        if self.success_counter >= self.success_threshold:
            self.success_counter = 0
            self.goal_index = (self.goal_index + 1) % self.total_phases
            self.goal = self._generate_goal()
            print(f"[GOAL] Switched to goal #{self.goal_index}: {self.goal}")
