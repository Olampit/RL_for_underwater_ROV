# environment.py

from pymavlink import mavutil
import numpy as np
import time
import subprocess
from joystick_input import FakeJoystick
import math
import random

from imu_reader import attitude_buffer, velocity_buffer

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

    def get_state(self):
        state = {}
        att_seq = attitude_buffer.get_all()
        if att_seq:
            yaws = np.array([abs(d["yawspeed"]) for _, d in att_seq if "yawspeed" in d])
            pitchs = np.array([abs(d["pitchspeed"]) for _, d in att_seq if "pitchspeed" in d])
            rolls = np.array([abs(d["rollspeed"]) for _, d in att_seq if "rollspeed" in d])
            if yaws.size > 0:
                state["yaw_mean"] = yaws.mean()
            if pitchs.size > 0:
                state["pitch_mean"] = pitchs.mean()
            if rolls.size > 0:
                state["roll_mean"] = rolls.mean()
        vel_seq = velocity_buffer.get_all()
        if vel_seq:
            vxs = np.array([v["vx"] for _, v in vel_seq])
            vys = np.array([v["vy"] for _, v in vel_seq])
            vzs = np.array([v["vz"] for _, v in vel_seq])
            state["vx_mean"] = vxs.mean()
            state["vy_mean"] = vys.mean()
            state["vz_mean"] = vzs.mean()
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
        px = round(random.uniform(-1.0, 1.0), 2)
        py = round(random.uniform(49.0, 51.0), 2)
        pz = round(random.uniform(9.5, 10.5), 2)
        quat = self.random_orientation_quat(max_angle_deg=15)
        qx, qy, qz, qw = quat["x"], quat["y"], quat["z"], quat["w"]
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
        self.joystick.next_episode()
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

    def _goal_to_state(self, goal):
        return {
            "yaw_mean": 0.0,
            "pitch_mean": 0.0,
            "roll_mean": 0.0,
            "vx_mean": goal["vx"]["mean"],
            "vy_mean": goal["vy"]["mean"],
            "vz_mean": goal["vz"]["mean"]
        }

    def compute_reward(self, state):
        goal = self.joystick.get_target()
        vx = state.get("vx_mean", 0.0)
        vy = state.get("vy_mean", 0.0)
        vz = state.get("vz_mean", 0.0)
        yaw = abs(state.get("yaw_mean", 0.0))
        pitch = abs(state.get("pitch_mean", 0.0))
        roll = abs(state.get("roll_mean", 0.0))

        def sharp_score(x, target, min_val=0.0, sigma=0.03):
            if x < min_val:
                return -1.0
            err = abs(x - target)
            return np.exp(-(err ** 2) / (2 * sigma ** 2))

        def angular_penalty(x, sigma):
            return 1.0 - np.exp(-x ** 2 / (2 * sigma ** 2))

        vx_score = sharp_score(vx, goal["vx"]["mean"], min_val=0.0, sigma=0.03)
        vy_score = np.exp(-(vy ** 2) / (2 * 0.015 ** 2))
        vz_score = np.exp(-(vz ** 2) / (2 * 0.015 ** 2))

        angle_penalty = (
            angular_penalty(yaw, 0.7) +
            angular_penalty(pitch, 0.7) +
            angular_penalty(roll, 0.7)
        )

        total = 4.0 * vx_score + 1.0 * vy_score + 1.0 * vz_score - 3.5 * angle_penalty
        total = 10 * np.tanh(total / 10.0)

        return {
            "total": total,
            "velocity_score": vx_score,
            "stability_score": vy_score + vz_score,
            "angular_score": -angle_penalty,
            "bonus": 0.0,
            "yaw_rate": yaw,
            "pitch_rate": pitch,
            "roll_rate": roll
        }
