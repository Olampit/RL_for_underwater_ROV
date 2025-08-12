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
    def compute_reward_2(self):
        CLIP = 100.0
        MAX_AGE = 0.1
        SHAPING_WEIGHT = 2.0

        def wrap(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        def huber(x, delta=0.1):
            abs_x = np.abs(x)
            quadratic = np.minimum(abs_x, delta)
            linear = abs_x - quadratic
            return 0.5 * quadratic**2 + delta * linear

        now = time.time()
        vel_seq = velocity_buffer.get_since(now - MAX_AGE, max_age=MAX_AGE)
        att_seq = attitude_buffer.get_since(now - MAX_AGE, max_age=MAX_AGE)
        goal_seq = goal_buffer.get_since(now - MAX_AGE, max_age=MAX_AGE)



        vel_values = [v for _, v in vel_seq]
        att_values = [a for _, a in att_seq]
        goal_values = [g for _, g in goal_seq]

        vel_penalties = []
        orientation_penalties = []
        spin_penalties = []
        improvement_terms = []

        N = min(len(vel_values), len(att_values), len(goal_values))

        for i in range(1, N):

            prev_vel = vel_values[i - 1]
            prev_att = att_values[i - 1]
            prev_goal = goal_values[i - 1]

            curr_vel = vel_values[i]
            curr_att = att_values[i]
            curr_goal = goal_values[i]

            prev_vel_vec = np.array([prev_vel.get("vx", 0.0), prev_vel.get("vy", 0.0), prev_vel.get("vz", 0.0)]) / 1.5
            curr_vel_vec = np.array([curr_vel.get("vx", 0.0), curr_vel.get("vy", 0.0), curr_vel.get("vz", 0.0)]) / 1.5
            prev_goal_vel = np.array([prev_goal.get("vx", 0.0), prev_goal.get("vy", 0.0), prev_goal.get("vz", 0.0)]) / 0.5
            curr_goal_vel = np.array([curr_goal.get("vx", 0.0), curr_goal.get("vy", 0.0), curr_goal.get("vz", 0.0)]) / 0.5

            prev_err = prev_vel_vec - prev_goal_vel
            curr_err = curr_vel_vec - curr_goal_vel

            vel_penalty = np.sum(huber(curr_err))
            vel_penalties.append(vel_penalty)

            delta_err = np.linalg.norm(prev_err) - np.linalg.norm(curr_err)
            improvement_terms.append(np.clip(delta_err, -1.0, 1.0) * SHAPING_WEIGHT)

            roll_error = wrap(curr_att.get("roll", 0.0) - curr_goal.get("roll", 0.0)) / np.pi
            pitch_error = wrap(curr_att.get("pitch", 0.0) - curr_goal.get("pitch", 0.0)) / np.pi
            yaw_error = wrap(curr_att.get("yaw", 0.0) - curr_goal.get("yaw", 0.0)) / np.pi

            orientation_error = np.array([roll_error, pitch_error, yaw_error])
            orientation_penalty = np.sum(huber(orientation_error))
            orientation_penalties.append(orientation_penalty)

            rollspeed = curr_att.get("rollspeed", 0.0)
            pitchspeed = curr_att.get("pitchspeed", 0.0)
            yawspeed = curr_att.get("yawspeed", 0.0)
            spin = np.array([rollspeed, pitchspeed, yawspeed])
            spin_penalty = np.sum(huber(spin))
            spin_penalties.append(spin_penalty)

        vel_term = np.mean(vel_penalties)
        ori_term = np.mean(orientation_penalties)
        spin_term = np.mean(spin_penalties)
        shaping_bonus = np.mean(improvement_terms)

        total = -vel_term - ori_term - 3.0 * spin_term + shaping_bonus
        total = total
        total = np.clip(total, -CLIP, CLIP)

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

        return {
            "total": total,
            "velocity_alignment": -vel_term,
            "orientation_alignment": -ori_term,
            "shaping_bonus": shaping_bonus,
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
