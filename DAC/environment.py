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
            yawspeeds = np.array([d["yawspeed"] for _, d in att_seq if "yawspeed" in d])
            pitchspeeds = np.array([d["pitchspeed"] for _, d in att_seq if "pitchspeed" in d])
            rollspeeds = np.array([d["rollspeed"] for _, d in att_seq if "rollspeed" in d])

            if len(yawspeeds) >= 2:
                state["yaw_rate"] = yawspeeds[-1]
                state["yaw_rate_diff"] = yawspeeds[-1] - yawspeeds[-2]
                state["yaw_rate_std"] = np.std(np.diff(yawspeeds))
            if len(pitchspeeds) >= 2:
                state["pitch_rate"] = pitchspeeds[-1]
                state["pitch_rate_diff"] = pitchspeeds[-1] - pitchspeeds[-2]
                state["pitch_rate_std"] = np.std(np.diff(pitchspeeds))
            if len(rollspeeds) >= 2:
                state["roll_rate"] = rollspeeds[-1]
                state["roll_rate_diff"] = rollspeeds[-1] - rollspeeds[-2]
                state["roll_rate_std"] = np.std(np.diff(rollspeeds))

        vel_seq = velocity_buffer.get_all()
        if vel_seq:
            vxs = np.array([v["vx"] for _, v in vel_seq])
            vys = np.array([v["vy"] for _, v in vel_seq])
            vzs = np.array([v["vz"] for _, v in vel_seq])

            if len(vxs) >= 2:
                state["vx"] = vxs[-1]
                state["vx_diff"] = vxs[-1] - vxs[-2]
                state["vx_std"] = np.std(np.diff(vxs))
            if len(vys) >= 2:
                state["vy"] = vys[-1]
                state["vy_diff"] = vys[-1] - vys[-2]
                state["vy_std"] = np.std(np.diff(vys))
            if len(vzs) >= 2:
                state["vz"] = vzs[-1]
                state["vz_diff"] = vzs[-1] - vzs[-2]
                state["vz_std"] = np.std(np.diff(vzs))

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
        quat = self.random_orientation_quat(max_angle_deg=0)
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
            "vx": goal["vx"]["mean"],
            "vx_diff": 0.0,
            "vx_std": 0.0,
            "vy": goal["vy"]["mean"],
            "vy_diff": 0.0,
            "vy_std": 0.0,
            "vz": goal["vz"]["mean"],
            "vz_diff": 0.0,
            "vz_std": 0.0,
            "yaw_rate": goal["yaw_rate"]["mean"],
            "yaw_rate_diff": 0.0,
            "yaw_rate_std": 0.0,
            "pitch_rate": goal["pitch_rate"]["mean"],
            "pitch_rate_diff": 0.0,
            "pitch_rate_std": 0.0,
            "roll_rate": goal["roll_rate"]["mean"],
            "roll_rate_diff": 0.0,
            "roll_rate_std": 0.0
        }


    def compute_reward(self, state):
        goal = self.joystick.get_target()

        V_MAX = 1.0
        R_MAX = 2.0
        SMOOTH_W = 0.5
        MULT = 5

        vx = state.get("vx", 0.0)
        vy = state.get("vy", 0.0)
        vz = state.get("vz", 0.0)
        yaw_rate = abs(state.get("yaw_rate", 0.0))
        pitch_rate = abs(state.get("pitch_rate", 0.0))
        roll_rate = abs(state.get("roll_rate", 0.0))

        vx_error = abs(vx - goal["vx"]["mean"]) / V_MAX
        vy_error = abs(vy - goal["vy"]["mean"]) / V_MAX
        vz_error = abs(vz - goal["vz"]["mean"]) / V_MAX

        yaw_error = abs(yaw_rate - goal["yaw_rate"]["mean"]) / R_MAX
        pitch_error = abs(pitch_rate - goal["pitch_rate"]["mean"]) / R_MAX
        roll_error = abs(roll_rate - goal["roll_rate"]["mean"]) / R_MAX

        vx_score = -vx_error - SMOOTH_W * state.get("vx_std", 0.0)
        vy_score = -vy_error - SMOOTH_W * state.get("vy_std", 0.0)
        vz_score = -vz_error - SMOOTH_W * state.get("vz_std", 0.0)

        yaw_score = -yaw_error - SMOOTH_W * state.get("yaw_rate_std", 0.0)
        pitch_score = -pitch_error - SMOOTH_W * state.get("pitch_rate_std", 0.0)
        roll_score = -roll_error - SMOOTH_W * state.get("roll_rate_std", 0.0)

        total = (vx_score + vy_score + vz_score + yaw_score + pitch_score + roll_score) * MULT

        return {
            "total": total,
            "vx_score": vx_score,
            "vy_score": vy_score,
            "vz_score": vz_score,
            "roll_score": roll_score,
            "pitch_score": pitch_score,
            "yaw_score": yaw_score,
            "yaw_rate": yaw_rate,
            "pitch_rate": pitch_rate,
            "roll_rate": roll_rate
        }



    def is_terminal(self, state):
        return False