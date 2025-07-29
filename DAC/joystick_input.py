import numpy as np
import random

class FakeJoystick:
    def __init__(self, seed=42, total_phases=100000, transition_phase=5000, evaluation_mode=False):
        self.episode = 0
        self.total_phases = total_phases
        self.transition_phase = transition_phase  # NEW: when to start orientation goals
        self.evaluation_mode = evaluation_mode
        self.manual_goal = None
        random.seed(seed)

        self.velocity_schedule = self._generate_velocity_schedule()
        self.orientation_schedule = self._generate_orientation_schedule()
        self.velocity_index = 0
        self.orientation_index = 0
        self.goal = self._generate_goal()

        # Independent success tracking
        self.success_counter_v = 0
        self.success_counter_r = 0
        self.success_threshold = 1000
        self.error_threshold_v = 0.05
        self.error_threshold_r = 0.0 #! retablish higher or wont work

    def _generate_velocity_schedule(self):
        schedule = []

        for _ in range(self.total_phases):
            # Always include vz as 0.0
            goal = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
            
            # Only randomly activate vx or vy
            active = random.sample(["vx", "vy"], k=1)  # set k=2 if you want combos
            for axis in active:
                goal[axis] = random.choice([0.1, 0.2, 0.3, 0.4, 0.5]) * random.choice([1, -1])
            
            schedule.append(goal)

        # Add final zero goal
        schedule.append({"vx": 0.0, "vy": 0.0, "vz": 0.0})
        return schedule


    def _generate_orientation_schedule(self):
        directions = ["roll", "pitch", "yaw"]
        schedule = []

        for i in range(self.total_phases):
            goal = {d: 0.0 for d in directions}
            if i >= self.transition_phase:
                active = random.sample(directions, k=1)
                for axis in active:
                    goal[axis] = random.uniform(-np.pi / 2, np.pi / 2)
            schedule.append(goal)

        schedule.append({d: 0.0 for d in directions})
        return schedule



    def _generate_goal(self):
        if self.evaluation_mode and self.manual_goal is not None:
            return {k: {"mean": v, "std": 0.0} for k, v in self.manual_goal.items()}
        else:
            v_goal = self.velocity_schedule[self.velocity_index]
            r_goal = self.orientation_schedule[self.orientation_index]
            combined = {**v_goal, **r_goal}
            return {k: {"mean": v, "std": 0.0} for k, v in combined.items()}

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

    def _switch_velocity_goal(self):
        self.velocity_index = (self.velocity_index + 1) % self.total_phases
        self.goal = self._generate_goal()
        print(f"[GOAL] Velocity updated: {self.velocity_schedule[self.velocity_index]}")

    def _switch_orientation_goal(self):
        self.orientation_index = (self.orientation_index + 1) % self.total_phases
        self.goal = self._generate_goal()
        print(f"[GOAL] Orientation updated: {self.orientation_schedule[self.orientation_index]}")

    def update_success_tracking(self, reward_components):
        if self.evaluation_mode:
            return

        # Extract per-axis errors
        vx_e = abs(reward_components.get("vx_error", 1.0))
        vy_e = abs(reward_components.get("vy_error", 1.0))
        vz_e = abs(reward_components.get("vz_error", 1.0))
        yaw_e = abs(reward_components.get("yaw_error", 1.0))
        pitch_e = abs(reward_components.get("pitch_error", 1.0))
        roll_e = abs(reward_components.get("roll_error", 1.0))

        # === NEW: only consider errors for active goal axes ===
        active_v_axes = ["vx", "vy"]
        active_r_axes = ["yaw", "pitch", "roll"]

        v_errors = [abs(reward_components.get(f"{k}_error", 1.0)) for k in active_v_axes]
        r_errors = [abs(reward_components.get(f"{k}_error", 1.0)) for k in active_r_axes]

        v_error_max = max(v_errors) if v_errors else 1.0
        r_error_max = max(r_errors) if r_errors else 1.0

        # === Update exponential moving average of errors ===
        if not hasattr(self, "error_avg_v"):
            self.error_avg_v = 1.0
        if not hasattr(self, "error_avg_r"):
            self.error_avg_r = 1.0

        self.error_avg_v = 0.9 * self.error_avg_v + 0.1 * v_error_max
        self.error_avg_r = 0.9 * self.error_avg_r + 0.1 * r_error_max

        # === Track success over time ===
        if self.error_avg_v < self.error_threshold_v:
            self.success_counter_v += 1
        else:
            self.success_counter_v = max(0, self.success_counter_v - 1)

        if self.error_avg_r < self.error_threshold_r:
            self.success_counter_r += 1
        else:
            self.success_counter_r = max(0, self.success_counter_r - 1)

        if self.success_counter_v >= self.success_threshold:
            self.success_counter_v = 0
            self._switch_velocity_goal()

        if self.success_counter_r >= self.success_threshold:
            self.success_counter_r = 0
            self._switch_orientation_goal()

