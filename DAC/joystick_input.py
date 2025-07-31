import numpy as np
import random

class FakeJoystick:
    """
    This class simulates a joystick or human input interface that generates 
    a sequence of target velocities and orientations for the ROV to follow.

    It's used during training to provide dynamic, automatic goals.
    """
    
    def __init__(self, seed=42, total_phases=100000, transition_phase=5000, evaluation_mode=False):
        """
        Initializes the FakeJoystick with parameters controlling its behavior.

        Args:
            seed (int): Random seed for reproducibility.
            total_phases (int): Number of goal phases before repeating.
            transition_phase (int): When to start issuing orientation goals.
            evaluation_mode (bool): If True, uses a fixed manual goal (for testing).
        """
        
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

        # Success tracking for both velocity and orientation goals
        self.success_counter_v = 0
        self.success_counter_r = 0
        self.success_threshold = 1000 # Unused now, kept for backward compatibility
        self.error_threshold_v = 0.05 # Tolerance for velocity error
        self.error_threshold_r = 0.0 # Tolerance for orientation error #! retablish higher when using orientation goals or wont work 

    def _generate_velocity_schedule(self):
        """
        Randomly generates a list of velocity targets (vx, vy, vz) for each phase.
        Only one axis is activated per goal for training simplicity.
        """
        
        schedule = []

        for _ in range(self.total_phases):
            # Always include vz as 0.0
            goal = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
            
            # Only randomly activate vx or vy
            active = random.sample(["vx", "vy"], k=1)  # set k=2 if you want combos
            for axis in active:
                goal[axis] = random.choice([0.1, 0.2, 0.3, 0.4, 0.5]) * random.choice([1, -1])
            
            schedule.append(goal)

        schedule.append({"vx": 0.0, "vy": 0.0, "vz": 0.0}) # Final stop goal    
        return schedule


    def _generate_orientation_schedule(self):
        """
        Randomly generates a list of orientation goals (roll, pitch, yaw) for each phase.
        Only activates after `transition_phase` has been reached.
        """
        
        directions = ["roll", "pitch", "yaw"]
        schedule = []

        for i in range(self.total_phases):
            goal = {d: 0.0 for d in directions}
            if i >= self.transition_phase:
                active = random.sample(directions, k=1)
                for axis in active:
                    goal[axis] = random.uniform(-np.pi / 2, np.pi / 2)
            schedule.append(goal)

        schedule.append({d: 0.0 for d in directions}) # Final neutral orientation
        return schedule



    def _generate_goal(self):
        """
        Returns a new goal by combining current velocity and orientation targets.
        In evaluation mode, returns the manual goal instead.
        """
        
        if self.evaluation_mode and self.manual_goal is not None:
            return {k: {"mean": v, "std": 0.0} for k, v in self.manual_goal.items()}
        else:
            v_goal = self.velocity_schedule[self.velocity_index]
            r_goal = self.orientation_schedule[self.orientation_index]
            combined = {**v_goal, **r_goal}
            return {k: {"mean": v, "std": 0.0} for k, v in combined.items()}

    def set_manual_goal(self, goal_dict):
        """
        Enables evaluation mode with a fixed manual goal (bypasses schedules).
        """
        
        self.manual_goal = goal_dict
        self.evaluation_mode = True
        self.goal = self._generate_goal()

    def enable_training_mode(self):
        """
        Switches back from evaluation to training mode (schedule-based goals).
        """
        
        self.evaluation_mode = False
        self.manual_goal = None
        self.goal = self._generate_goal()

    def get_target(self):
        """
        Returns the current goal (used by reward computation and control loop).
        """
        
        return self.goal

    def _switch_velocity_goal(self):
        """
        Advances to the next velocity goal in the schedule and regenerates the full goal.
        """
        
        self.velocity_index = (self.velocity_index + 1) % self.total_phases
        self.goal = self._generate_goal()
        print(f"[GOAL] Velocity updated: {self.velocity_schedule[self.velocity_index]}")

    def _switch_orientation_goal(self):
        """
        Advances to the next orientation goal in the schedule and regenerates the full goal.
        """
        
        self.orientation_index = (self.orientation_index + 1) % self.total_phases
        self.goal = self._generate_goal()
        print(f"[GOAL] Orientation updated: {self.orientation_schedule[self.orientation_index]}")

    def update_success_tracking(self, reward_components):
        """
        Monitors how well the ROV is matching its current goal. If it's been successful
        on active axes for 20 consecutive steps, the goal is switched.

        Args:
            reward_components (dict): Output from the reward function, including *_error values.
        """
        
        if self.evaluation_mode:
            return # Do not update goal in evaluation mode


        # --- ACTIVE AXES ONLY ---
        # Axes we care about
        active_v_axes = ["vx", "vy"]
        active_r_axes = ["yaw", "pitch", "roll"]

        # Velocity success tracking
        for axis in active_v_axes:
            if abs(self.goal[axis]["mean"]) < 1e-4:
                continue
            error = abs(reward_components.get(f"{axis}_error", 1.0))
            if error < self.error_threshold_v:
                self.success_counter_v += 1
            else:
                self.success_counter_v = 0  # must be consecutive!

        # Orientation success tracking
        for axis in active_r_axes:
            if abs(self.goal[axis]["mean"]) < 1e-4:
                continue
            error = abs(reward_components.get(f"{axis}_error", 1.0))
            if error < self.error_threshold_r:
                self.success_counter_r += 1
            else:
                self.success_counter_r = 0

        # --- Switch when 20 consecutive successes observed ---
        if self.success_counter_v >= 20:
            self.success_counter_v = 0
            self._switch_velocity_goal()

        if self.success_counter_r >= 20:
            self.success_counter_r = 0
            self._switch_orientation_goal()

