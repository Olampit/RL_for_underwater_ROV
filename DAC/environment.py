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

    def get_state(self, time_before_action):
        state = {}
        MAX_AGE = 0.1
        start_time = time_before_action

        raw_seq = raw_buffer.get_since(start_time, max_age=MAX_AGE)
        att_seq = attitude_buffer.get_since(start_time, max_age=MAX_AGE)
        goal_seq = goal_buffer.get_since(start_time, max_age=MAX_AGE)

        # Extract goal values
        if not goal_seq:
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

        # Extract raw IMU acceleration
        if raw_seq:
            state["ax"] = sum(r.get("ax", 0.0) for _, r in raw_seq) / len(raw_seq)
            state["ay"] = sum(r.get("ay", 0.0) for _, r in raw_seq) / len(raw_seq)
            state["az"] = sum(r.get("az", 0.0) for _, r in raw_seq) / len(raw_seq)
        else:
            state["ax"] = state["ay"] = state["az"] = 0.0

        # Extract orientation (RPY)
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
        time_before_reset = time.time()

        px, py, pz = 0, 5000, 50

        odom_seq = velocity_buffer.get_last_n(1)
        if odom_seq:
            _, last_data = odom_seq[0]
            qx = last_data.get("qx", 0.0)
            qy = last_data.get("qy", 0.0)
            qz = last_data.get("qz", 0.0)
            qw = last_data.get("qw", 1.0)
        else:
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
        return self.get_state(time_before_reset)

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

    def compute_reward(self):
        CLIP = 100.0
        MAX_AGE = 0.1

        SCALE_VEL = 1.0
        SCALE_SPIN = 1.0
        COEFF = 1.0
        SPIN_WEIGHT = 1.0
        STD_WEIGHT = 0.5
        STILLNESS_BONUS = 1.0
        SPIN_THRESHOLD = 0.05

        def wrap(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        def penalty(x, scale):
            return -COEFF * np.log1p((abs(x) / scale) ** 2)

        def compute_std(values, key):
            data = [v.get(key, 0.0) for v in values]
            return np.std(data) if data else 0.0

        now = time.time()
        vel_seq = velocity_buffer.get_since(now - MAX_AGE, max_age=MAX_AGE)
        att_seq = attitude_buffer.get_since(now - MAX_AGE, max_age=MAX_AGE)
        goal_seq = goal_buffer.get_since(now - MAX_AGE, max_age=MAX_AGE)

        if not vel_seq or not att_seq or not goal_seq:
            print("missing")
            return {"total": -CLIP, "reason": "missing data"}

        vel_values = [v for _, v in vel_seq]
        att_values = [a for _, a in att_seq]
        goal_values = [g for _, g in goal_seq]

        # Sum of squared goal values
        goal_vx = sum(g.get("vx", 0.0) ** 2 for g in goal_values)
        goal_vy = sum(g.get("vy", 0.0) ** 2 for g in goal_values)
        goal_vz = sum(g.get("vz", 0.0) ** 2 for g in goal_values)
        goal_yaw = sum(g.get("yaw", 0.0) ** 2 for g in goal_values)
        goal_pitch = sum(g.get("pitch", 0.0) ** 2 for g in goal_values)
        goal_roll = sum(g.get("roll", 0.0) ** 2 for g in goal_values)

        # Sum of squared observed values
        vx = sum(v.get("vx", 0.0) ** 2 for v in vel_values)
        vy = sum(v.get("vy", 0.0) ** 2 for v in vel_values)
        vz = sum(v.get("vz", 0.0) ** 2 for v in vel_values)

        yaw = sum(a.get("yaw", 0.0) ** 2 for a in att_values)
        pitch = sum(a.get("pitch", 0.0) ** 2 for a in att_values)
        roll = sum(a.get("roll", 0.0) ** 2 for a in att_values)

        # Angular rates squared
        mean_yawspeed = sum(abs(a.get("yawspeed", 0.0)) ** 2 for a in att_values)
        mean_pitchspeed = sum(abs(a.get("pitchspeed", 0.0)) ** 2 for a in att_values)
        mean_rollspeed = sum(abs(a.get("rollspeed", 0.0)) ** 2 for a in att_values)

        # Std penalties
        vx_std = compute_std(vel_values, "vx")
        vy_std = compute_std(vel_values, "vy")
        vz_std = compute_std(vel_values, "vz")
        yaw_std = compute_std(att_values, "yaw")
        pitch_std = compute_std(att_values, "pitch")
        roll_std = compute_std(att_values, "roll")

        std_penalty = -STD_WEIGHT * (
            vx_std + vy_std + vz_std +
            yaw_std + pitch_std + roll_std
        )

        # Angular spin penalties
        yaw_spin = penalty(mean_yawspeed, SCALE_SPIN)
        pitch_spin = penalty(mean_pitchspeed, SCALE_SPIN)
        roll_spin = penalty(mean_rollspeed, SCALE_SPIN)

        # Angular alignment errors
        yaw_error = abs(wrap(yaw - goal_yaw))
        pitch_error = abs(wrap(pitch - goal_pitch))
        roll_error = abs(wrap(roll - goal_roll))

        yaw_alignment = penalty(yaw_error, SCALE_SPIN)
        pitch_alignment = penalty(pitch_error, SCALE_SPIN)
        roll_alignment = penalty(roll_error, SCALE_SPIN)

        yaw_score = yaw_alignment + yaw_spin * SPIN_WEIGHT
        pitch_score = pitch_alignment + pitch_spin * SPIN_WEIGHT
        roll_score = roll_alignment + roll_spin * SPIN_WEIGHT

        # Velocity tracking error
        vx_error = vx - goal_vx
        vy_error = vy - goal_vy
        vz_error = vz - goal_vz

        vx_score = penalty(vx_error, SCALE_VEL)
        vy_score = penalty(vy_error, SCALE_VEL)
        vz_score = penalty(vz_error, SCALE_VEL)

        # Total reward
        total = (
            vx_score +
            vy_score +
            vz_score +
            yaw_score +
            pitch_score +
            roll_score +
            std_penalty
        )

        # Stillness bonus when goal is to be still and spin is low
        if (
            abs(goal_vx) < 0.1 and abs(goal_vy) < 0.1 and abs(goal_vz) < 0.1 and
            all(spin < SPIN_THRESHOLD for spin in [mean_yawspeed, mean_pitchspeed, mean_rollspeed])
        ):
            total += STILLNESS_BONUS

        total = np.clip(total, -CLIP, CLIP)

        return {
            "total": total,
            "vx_score": vx_score,
            "vy_score": vy_score,
            "vz_score": vz_score,
            "yaw_score": yaw_score,
            "pitch_score": pitch_score,
            "roll_score": roll_score,
            "std_penalty": std_penalty,
            "yaw_spin": yaw_spin,
            "pitch_spin": pitch_spin,
            "roll_spin": roll_spin,
            "vx_std": vx_std,
            "vy_std": vy_std,
            "vz_std": vz_std,
            "yaw_std": yaw_std,
            "pitch_std": pitch_std,
            "roll_std": roll_std,
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
            "vx_error": vx_error,
            "vy_error": vy_error,
            "vz_error": vz_error,
            "roll_error": roll_error,
            "pitch_error": pitch_error,
            "yaw_error": yaw_error,
        }


    def is_terminal(self):
        return False
