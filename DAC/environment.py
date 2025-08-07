from pymavlink import mavutil
import numpy as np
import time
import subprocess
from joystick_input import FakeJoystick
import math
import random


# Buffers to hold history of velocity, attitude, and goal for short-term analysis. Raw is for accelerations
from imu_reader import attitude_buffer, velocity_buffer, goal_buffer, raw_buffer


# --------------------------
# PWM (Pulse Width Modulation) Constants
# --------------------------
SERVO_MIN = 1100     # Minimum PWM signal sent to ESC (full reverse)
SERVO_MAX = 1900     # Maximum PWM signal (full forward)
SERVO_IDLE = 1500    # Neutral signal 





RADIAN_SCALING = 57.2958  # Matches np.deg2rad(1), used here purely for scaling effect



# --------------------------
# Function: input_to_pwm
# Converts a normalized float [-1.0, 1.0] to PWM signal
# --------------------------
def input_to_pwm(value):
    if abs(value) < 0.05:
        return SERVO_IDLE # Deadzone for small noise; don't move motor
    pwm = SERVO_IDLE + (value * 400)
    return int(max(SERVO_MIN, min(SERVO_MAX, pwm)))


# --------------------------
# Class: ROVEnvironment
# Represents the control interface to the ROV
# Used to apply actions, gather states, and reset simulation
# --------------------------
class ROVEnvironment:
    def __init__(self, action_map, connection):
        self.action_map = action_map        # Deadzone for small noise; don't move motor
        self.connection = connection        # MAVLink connection object
        self.joystick = FakeJoystick()      # Goal generator 
        self.target_velocity = self.joystick.get_target()


    # --------------------------
    # Function: apply_action
    # Converts an action index to motor commands and sends them via MAVLink
    # --------------------------
    def apply_action(self, action_idx):
        action = self.action_map[action_idx]
        for i in range(8):      # 8 motors
            motor_label = f"motor{i+1}"
            thrust = action.get(motor_label, 0.0)
            pwm = input_to_pwm(thrust)
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,  #Motors are considered servos by mavlink
                0,
                i + 1,
                pwm,
                0, 0, 0, 0, 0
            )


    # --------------------------
    # Function: get_state
    # Returns a state vector representing the ROV state from logs
    # Aggregates goal, acceleration, and orientation data
    # --------------------------
    def get_state(self, time_before_action):
        state = {}
        MAX_AGE = 0.1 # Seconds of history to look back 
        start_time = time_before_action

        # Get buffered data sequences from sensors and goals, check imu_reader.py if needed
        raw_seq = raw_buffer.get_since(start_time, max_age=MAX_AGE)
        att_seq = attitude_buffer.get_since(start_time, max_age=MAX_AGE)
        goal_seq = goal_buffer.get_since(start_time, max_age=MAX_AGE)

            # --- GOAL VALUES (target velocity/orientation) ---        
        if not goal_seq:
            # If no goal is found, default to zero goal
            for k in ["goal_vx", "goal_vy", "goal_vz", "goal_roll", "goal_pitch", "goal_yaw"]:
                state[k] = 0.0
        else:
            goal_values = [g for _, g in goal_seq]
            state["goal_vx"] = sum(g.get("vx", 0.0) for g in goal_values) / len(goal_values)
            state["goal_vy"] = sum(g.get("vy", 0.0) for g in goal_values) / len(goal_values)
            state["goal_vz"] = sum(g.get("vz", 0.0) for g in goal_values) / len(goal_values)
            state["goal_roll"] = sum(g.get("roll", 0.0) for g in goal_values) / len(goal_values)
            state["goal_pitch"] = sum(g.get("pitch", 0.0) for g in goal_values) / len(goal_values)
            state["goal_yaw"] = sum(g.get("yaw", 0.0) for g in goal_values) / len(goal_values)

        
        # --- RAW IMU ACCELERATIONS (ax, ay, az) ---
        if raw_seq:
            state["ax"] = sum(r.get("ax", 0.0) for _, r in raw_seq) / len(raw_seq)
            state["ay"] = sum(r.get("ay", 0.0) for _, r in raw_seq) / len(raw_seq)
            state["az"] = sum(r.get("az", 0.0) for _, r in raw_seq) / len(raw_seq)
        else:
            state["ax"] = state["ay"] = state["az"] = 0.0


        # --- ORIENTATION ESTIMATES (roll, pitch, yaw) ---
        if att_seq:
            rolls = [a.get("roll", 0.0) for _, a in att_seq]
            pitches = [a.get("pitch", 0.0) for _, a in att_seq]
            yaws = [a.get("yaw", 0.0) for _, a in att_seq]

            state["roll"] = sum(rolls) / len(rolls)
            state["pitch"] = sum(pitches) / len(pitches)
            state["yaw"] = sum(yaws) / len(yaws)
        else:
            state["roll"] = state["pitch"] = state["yaw"] = 0.0

        return state

    # --------------------------
    # Function: random_orientation_quat
    # Returns a random quaternion within a max tilt angle
    # Used for random resets or perturbations
    # --------------------------
    def random_orientation_quat(self, max_angle_deg=15):
        max_angle_rad = math.radians(max_angle_deg)
        roll = random.uniform(-max_angle_rad, max_angle_rad)
        pitch = random.uniform(-max_angle_rad, max_angle_rad)
        yaw = random.uniform(-math.pi, math.pi)
        
        # Convert RPY to quaternion using classic formulas
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        return {
            "x": sr * cp * cy - cr * sp * sy,
            "y": cr * sp * cy + sr * cp * sy,
            "z": cr * cp * sy - sr * sp * cy,
            "w": cr * cp * cy + sr * sp * sy,
        }


    # --------------------------
    # Function: reset
    # Respawns the robot at a fixed pose using ROS2 service call
    # Used at beginning of episodes to ensure consistent training
    # --------------------------
    def reset(self):
        time_before_reset = time.time()

        px, py, pz = 0, 5000, 70  # Fixed position in simulation world

        odom_seq = velocity_buffer.get_last_n(1)
        if odom_seq:
            _, last_data = odom_seq[0]
            qx = last_data.get("qx", 0.0)
            qy = last_data.get("qy", 0.0)
            qz = last_data.get("qz", 0.0)
            qw = last_data.get("qw", 1.0)
        else:
            qx, qy, qz, qw = 0.0, 0.0, np.sqrt(2)/2, np.sqrt(2)/2 # Default facing down

        cmd = [
            "ros2", "service", "call",
            "/stonefish_ros2/respawn_robot",
            "stonefish_ros2/srv/Respawn",
            f"""{{name: 'bluerov',
            origin: {{
                position: {{x: {px}, y: {py}, z: {pz}}},
                orientation: {{x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}, w: {qw:.6f}}}
            }}}}"""
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self.get_state(time_before_reset)


    # --------------------------
    # Function: stop_motors
    # Sends neutral PWM signals to all motors to ensure complete stop
    # --------------------------
    def stop_motors(self, connection):
        for servo in range(1, 9):
            connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                servo,
                1500,
                0, 0, 0, 0, 0
            )
            

    # --------------------------
    # Function: compute_reward
    # Computes the score that we want to give the state we are in.
    # --------------------------
    def compute_reward(self):
        CLIP = 100.0  # Max reward cap
        MAX_AGE = 0.1  # Time window in seconds

        # ------------- Helpers -------------
        def normalize(vec):
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                return np.zeros_like(vec)
            return vec / norm

        def dot_alignment(vec1, vec2):
            v1 = normalize(vec1)
            v2 = normalize(vec2)
            return float(np.dot(v1, v2))

        def wrap(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        # ------------- Extract Buffered Sequences -------------
        now = time.time()
        vel_seq = velocity_buffer.get_since(now - MAX_AGE, max_age=MAX_AGE)
        att_seq = attitude_buffer.get_since(now - MAX_AGE, max_age=MAX_AGE)
        goal_seq = goal_buffer.get_since(now - MAX_AGE, max_age=MAX_AGE)

        if not vel_seq or not att_seq or not goal_seq:
            return {"total": -CLIP, "reason": "missing data"}

        vel_values = [v for _, v in vel_seq]
        att_values = [a for _, a in att_seq]
        goal_values = [g for _, g in goal_seq]

        # ------------- Per-Sample Reward Aggregation -------------
        velocity_scores = []
        orientation_scores = []

        for vel, att, goal in zip(vel_values, att_values, goal_values):
            vel_vec = np.array([
                vel.get("vx", 0.0),
                vel.get("vy", 0.0),
                vel.get("vz", 0.0),
            ])
            goal_vel_vec = np.array([
                goal.get("vx", 0.0),
                goal.get("vy", 0.0),
                goal.get("vz", 0.0),
            ])

            att_vec = np.array([
                att.get("roll", 0.0),
                att.get("pitch", 0.0),
                att.get("yaw", 0.0),
            ])
            goal_att_vec = np.array([
                goal.get("roll", 0.0),
                goal.get("pitch", 0.0),
                goal.get("yaw", 0.0),
            ])

            vel_align = dot_alignment(vel_vec, goal_vel_vec)
            att_align = dot_alignment(att_vec, goal_att_vec)

            velocity_scores.append(vel_align)
            orientation_scores.append(att_align)

        # ------------- Reward Composition -------------
        velocity_penalty = sum([(-1.0 + s) ** 2 for s in velocity_scores]) #be careful that this isnt getting smaller with the **2 then it should be
        orientation_penalty = sum([(-1.0 + s) ** 2 for s in orientation_scores])
        total = -velocity_penalty - orientation_penalty
        total = total/100
        total = np.clip(total, -CLIP, CLIP)
        

        # ------------- Latest Observation for Debug Info -------------
        vx = vel_values[-1].get("vx", 0.0)
        vy = vel_values[-1].get("vy", 0.0)
        vz = vel_values[-1].get("vz", 0.0)

        yaw = att_values[-1].get("yaw", 0.0)
        pitch = att_values[-1].get("pitch", 0.0)
        roll = att_values[-1].get("roll", 0.0)

        goal_vx = goal_values[-1].get("vx", 0.0)
        goal_vy = goal_values[-1].get("vy", 0.0)
        goal_vz = goal_values[-1].get("vz", 0.0)

        goal_yaw = goal_values[-1].get("yaw", 0.0)
        goal_pitch = goal_values[-1].get("pitch", 0.0)
        goal_roll = goal_values[-1].get("roll", 0.0)

        # ------------- Return Reward and Debug Info -------------
        return {
            "total": total,
            "velocity_alignment": velocity_scores[-1] - 1.0,
            "orientation_alignment": orientation_scores[-1] - 1.0,

            "goal_vx": goal_vx,
            "goal_vy": goal_vy,
            "goal_vz": goal_vz,
            "goal_yaw": goal_yaw,
            "goal_pitch": goal_pitch,
            "goal_roll": goal_roll,

            "vx": vx,
            "vy": vy,
            "vz": vz,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,

            "vx_error": vx - goal_vx,
            "vy_error": vy - goal_vy,
            "vz_error": vz - goal_vz,
            "yaw_error": wrap(yaw - goal_yaw),
            "pitch_error": wrap(pitch - goal_pitch),
            "roll_error": wrap(roll - goal_roll),
        }





    def is_terminal(self):
        # as there are no terminal states currently, this thing is always false
        return False
