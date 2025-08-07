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
        self.changes_counter_v = 0
        self.success_counter_v = 0
        self.success_counter_r = 0
        self.success_threshold = 1000 # Unused now, kept for backward compatibility
        self.error_threshold_v = 0.03 # Tolerance for velocity error #! was 0.05, but too many where accepted
        self.error_threshold_r = 0.01 # Tolerance for orientation error #! retablish higher when using orientation goals or wont work 

    def _generate_velocity_schedule(self):
        """
        Generates a more varied and exploration-like schedule of velocity goals.
        Multiple axes can be active with realistic random speeds.
        """
        schedule = []

        for _ in range(self.total_phases):
            goal = {
                "vx": np.random.choice([0.0, np.random.uniform(-0.5, 0.5)]),
                "vy": np.random.choice([0.0, np.random.uniform(-0.5, 0.5)]),
                "vz": np.random.choice([0.0, np.random.uniform(-0.3, 0.3)])  # less powerful z maybe
            }

            # Optional: Avoid all-zero goal
            if goal["vx"] == 0.0 and goal["vy"] == 0.0 and goal["vz"] == 0.0:
                axis = random.choice(["vx", "vy", "vz"])
                goal[axis] = np.random.uniform(-0.5, 0.5)

            schedule.append(goal)

        # Final "stop" goal
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
                    goal[axis] = random.uniform(-np.pi / 6, np.pi / 6)  # ~±30°
            schedule.append(goal)

        schedule.append({d: 0.0 for d in directions})
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


    def switch_goal_randomly(self):
        """
        Updates the goal with small deltas to simulate realistic joystick movement.
        Prevents abrupt changes like full-forward to full-backward instantly.
        Handles both float and dict-based goal formats.
        """
        MAX_DELTA_LINEAR = 0.2
        MAX_DELTA_ANGULAR = np.pi / 30

        if self.goal is None:
            self.goal = {
                "vx": 0.0, "vy": 0.0, "vz": 0.0,
                "yaw": 0.0, "pitch": 0.0, "roll": 0.0
            }

        def get_mean(val):
            if isinstance(val, dict):
                return val.get("mean", 0.0)
            return val

        self.goal = {
            "vx": np.clip(get_mean(self.goal.get("vx", 0.0)) + np.random.uniform(-MAX_DELTA_LINEAR, MAX_DELTA_LINEAR), -0.5, 0.5),
            "vy": 0, #np.clip(get_mean(self.goal.get("vy", 0.0)) + np.random.uniform(-MAX_DELTA_LINEAR, MAX_DELTA_LINEAR), -0.5, 0.5),
            "vz": 0, #np.clip(get_mean(self.goal.get("vz", 0.0)) + np.random.uniform(-MAX_DELTA_LINEAR, MAX_DELTA_LINEAR), -0.5, 0.5),
            "yaw":0, # np.clip(get_mean(self.goal.get("yaw", 0.0)) + np.random.uniform(-MAX_DELTA_ANGULAR, MAX_DELTA_ANGULAR), -np.pi, np.pi),
            "pitch":0, # np.clip(get_mean(self.goal.get("pitch", 0.0)) + np.random.uniform(-MAX_DELTA_ANGULAR, MAX_DELTA_ANGULAR), -np.pi/6, np.pi/6),
            "roll":0,# np.clip(get_mean(self.goal.get("roll", 0.0)) + np.random.uniform(-MAX_DELTA_ANGULAR, MAX_DELTA_ANGULAR), -np.pi/6, np.pi/6)
        }

        
    def update_success_tracking(self, reward_components):
        """
        Monitors how well the ROV is matching its current goal. If it's been successful
        on all active axes for 20 consecutive steps, the goal is switched.

        Args:
            reward_components (dict): Output from the reward function, including *_error values.
        """
        if self.evaluation_mode:
            return  # Do not update goal in evaluation mode

        # Define axes of interest (always checked)
        active_v_axes = ["vx", "vy"]
        active_r_axes = ["yaw", "pitch", "roll"]

        # --- Velocity success ---
        v_errors = []
        for axis in active_v_axes:
            err = reward_components.get(f"{axis}_error", None)
            if err is None:
                v_success = False
                break
            v_errors.append(abs(err))
        else:
            v_success = all(e < self.error_threshold_v for e in v_errors)

        # --- Orientation success ---
        r_errors = []
        for axis in active_r_axes:
            err = reward_components.get(f"{axis}_error", None)
            if err is None:
                r_success = False
                break
            r_errors.append(abs(err))
        else:
            r_success = all(e < self.error_threshold_r for e in r_errors)

        # --- Update counters ---
        if v_success:
            self.success_counter_v += 1
        else:
            self.success_counter_v = 0

        if r_success:
            self.success_counter_r += 1
        else:
            self.success_counter_r = 0

        # --- Log if needed ---
        if v_success and self.success_counter_v>100:
            print(f"[TRACKING] Velocity step successful ({self.success_counter_v}/200). This is the {self.changes_counter_v}-th change")
        # if r_success:
        #     print(f"[TRACKING] Orientation step successful ({self.success_counter_r}/500)")

        # --- Trigger goal change ---
        if self.success_counter_v >= 200: #since we add 1 thrice every time for some reason, might as well multiply requirements by 3
            self.success_counter_v = 0
            self._switch_velocity_goal()
            self.changes_counter_v += 1 

        if self.success_counter_r >= 200:
            self.success_counter_r = 0
            self._switch_orientation_goal()
