# environment.py

from pymavlink import mavutil
import numpy as np
import time
import subprocess
from joystick_input import FakeJoystick
import math
import random

from imu_reader import attitude_buffer, velocity_buffer, goal_buffer, raw_buffer

SERVO_MIN = 1100
SERVO_MAX = 1900
SERVO_IDLE = 1500

def input_to_pwm(value):
    if abs(value) < 0.05:
        return SERVO_IDLE
    pwm = SERVO_IDLE + (value * 400)
    return int(max(SERVO_MIN, min(SERVO_MAX, pwm)))

class ROVEnvironment:
    def __init__(self, action_map, connection, latest_imu):
        self.action_map = action_map
        self.connection = connection
        self.latest_imu = latest_imu
        self.joystick = FakeJoystick()
        self.target_velocity = self.joystick.get_target()

    def apply_action(self, action_idx):
        action = self.action_map[action_idx]
        for i in range(8):
            motor_label = f"motor{i+1}"
            thrust = action.get(motor_label, 0.0)
            pwm = input_to_pwm(thrust)
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                i + 1,
                pwm,
                0, 0, 0, 0, 0
            )

    from imu_reader import attitude_buffer, raw_buffer, goal_buffer

    def get_state(self):
        """
        Returns a sensor-based, goal-conditioned state:
        - Linear acceleration from raw IMU
        - Orientation and angular velocity from ATTITUDE
        - Goal values from joystick (velocity and rotation rates)
        """
        state = {}

        # --- Raw linear acceleration ---
        raw_seq = raw_buffer.get_last_n(1)
        if raw_seq:
            _, r = raw_seq[0]
            state["ax"] = r.get("ax", 0.0)
            state["ay"] = r.get("ay", 0.0)
            state["az"] = r.get("az", 0.0)
        else:
            state["ax"] = state["ay"] = state["az"] = 0.0

        # --- Attitude and angular velocities ---
        att_seq = attitude_buffer.get_last_n(1)
        if att_seq:
            _, a = att_seq[0]
            state["roll"] = a.get("roll", 0.0)
            state["pitch"] = a.get("pitch", 0.0)
            state["yaw"] = a.get("yaw", 0.0)
            state["rollspeed"] = a.get("rollspeed", 0.0)
            state["pitchspeed"] = a.get("pitchspeed", 0.0)
            state["yawspeed"] = a.get("yawspeed", 0.0)
        else:
            for key in ["roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed"]:
                state[key] = 0.0

        # --- Goal from joystick ---
        goal_seq = goal_buffer.get_last_n(1)
        if goal_seq:
            _, g = goal_seq[0]
            state["goal_vx"] = g.get("vx", 0.0)
            state["goal_vy"] = g.get("vy", 0.0)
            state["goal_vz"] = g.get("vz", 0.0)
            state["goal_yaw_rate"] = g.get("yaw_rate", 0.0)
            state["goal_pitch_rate"] = g.get("pitch_rate", 0.0)
            state["goal_roll_rate"] = g.get("roll_rate", 0.0)
        else:
            for key in ["goal_vx", "goal_vy", "goal_vz", "goal_yaw_rate", "goal_pitch_rate", "goal_roll_rate"]:
                state[key] = 0.0

        return state

    



    def random_orientation_quat(self, max_angle_deg=15):
        max_angle_rad = math.radians(max_angle_deg)
        roll = random.uniform(-max_angle_rad, max_angle_rad)
        pitch = random.uniform(-max_angle_rad, max_angle_rad)
        yaw = random.uniform(-math.pi, math.pi)
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

    def reset(self):
        # px = round(random.uniform(-1.0, 1.0), 2)
        # py = round(random.uniform(4999.0, 5001.0), 2)
        # pz = round(random.uniform(39.0, 41.0), 2)
        
        px = 0
        py = 5000
        pz = 20
        
        # quat = self.random_orientation_quat(max_angle_deg=0)
        # qx, qy, qz, qw = quat["x"], quat["y"], quat["z"], quat["w"]
        
        # qx, qy, qz, qw = 0, 0, (np.sqrt(2))/2, (np.sqrt(2))/2
        
        odom_seq = velocity_buffer.get_last_n(1)
        if odom_seq:
            _, last_data = odom_seq[0]
            qx = last_data.get("qx", 0.0)
            qy = last_data.get("qy", 0.0)
            qz = last_data.get("qz", 0.0)
            qw = last_data.get("qw", 1.0)
        else:
            # Fallback in case buffer is empty
            qx, qy, qz, qw = 0.0, 0.0, np.sqrt(2)/2, np.sqrt(2)/2
            
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
        return self.get_state()

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



    def compute_reward(self, state):
        TRACKING_WEIGHT = 1.0
        STABILITY_WEIGHT = 0.01
        CLIP = 100.0

        # Errors
        vx_e = state["vx_error"]
        vy_e = state["vy_error"]
        vz_e = state["vz_error"]
        yaw_e = state["yaw_error"]
        pitch_e = state["pitch_error"]
        roll_e = state["roll_error"]

        # Scales and coefficients
        V_SCALE = 0.5
        R_SCALE = 1.0
        COEFF_V = 1.0
        COEFF_A = 1.0

        def shaped_penalty(err, scale, coeff):
            norm_err = err / scale
            return -coeff * np.log1p(norm_err**2)

        vx_score = shaped_penalty(vx_e, V_SCALE, COEFF_V)
        vy_score = shaped_penalty(vy_e, V_SCALE, COEFF_V)
        vz_score = shaped_penalty(vz_e, V_SCALE, COEFF_V)
        yaw_score = shaped_penalty(yaw_e, R_SCALE, COEFF_A)
        pitch_score = shaped_penalty(pitch_e, R_SCALE, COEFF_A)
        roll_score = shaped_penalty(roll_e, R_SCALE, COEFF_A)

        tracking_total = (vx_score + vy_score + vz_score + yaw_score + pitch_score + roll_score) * TRACKING_WEIGHT

        # Stability
        vel_seq = velocity_buffer.get_last_n(5)
        att_seq = attitude_buffer.get_last_n(5)

        vxs = np.array([v["vx"] for _, v in vel_seq])
        vys = np.array([v["vy"] for _, v in vel_seq])
        vzs = np.array([v["vz"] for _, v in vel_seq])
        yaws = np.array([a["yawspeed"] for _, a in att_seq])
        pitches = np.array([a["pitchspeed"] for _, a in att_seq])
        rolls = np.array([a["rollspeed"] for _, a in att_seq])

        vel_std = np.std(vxs) + np.std(vys) + np.std(vzs)
        att_std = np.std(yaws) + np.std(pitches) + np.std(rolls)

        stability_penalty = (vel_std + att_std) * STABILITY_WEIGHT

        total_reward = tracking_total - stability_penalty
        total_reward = np.clip(total_reward, -CLIP, CLIP)

        return {
            "total": total_reward,
            "vx_score": vx_score,
            "vy_score": vy_score,
            "vz_score": vz_score,
            "yaw_score": yaw_score,
            "pitch_score": pitch_score,
            "roll_score": roll_score,
            "tracking_total": tracking_total,
            "stability_penalty": -stability_penalty
        }







    def is_terminal(self, state):
        return False