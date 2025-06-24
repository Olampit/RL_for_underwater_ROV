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
                state["yaw_rate"] = float(np.mean(np.abs(yawspeeds)))
            if len(pitchspeeds) >= 2:
                state["pitch_rate"] = float(np.mean(np.abs(pitchspeeds)))
            if len(rollspeeds) >= 2:
                state["roll_rate"] = float(np.mean(np.abs(rollspeeds)))

        vel_seq = velocity_buffer.get_all()
        if vel_seq:
            vxs = np.array([v["vx"] for _, v in vel_seq])
            vys = np.array([v["vy"] for _, v in vel_seq])
            vzs = np.array([v["vz"] for _, v in vel_seq])

            if len(vxs) >= 2:
                state["vx"] = float(np.mean(np.abs(vxs)))
            if len(vys) >= 2:
                state["vy"] = float(np.mean(np.abs(vys)))
            if len(vzs) >= 2:
                state["vz"] = float(np.mean(np.abs(vzs)))

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
        py = round(random.uniform(4900.0, 5100.0), 2)
        pz = round(random.uniform(9.5, 10.5), 2)
        
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
            "vy": goal["vy"]["mean"],
            "vz": goal["vz"]["mean"],
            "yaw_rate": goal["yaw_rate"]["mean"],
            "pitch_rate": goal["pitch_rate"]["mean"],
            "roll_rate": goal["roll_rate"]["mean"]
        }


    def compute_reward(self, state):
        goal = self.joystick.get_target()

        V_MAX = 1.0
        R_MAX = 2.0
        BONUS = 0.5
        DECAY = 10.0
        MULT = 1.0

        def e(key): return state.get(key, 0.0)
        def g(key): return goal[key]["mean"]
        def bonus(err): return BONUS * np.exp(-DECAY * err)

        # Normalized errors
        vx_err = abs(e("vx") - g("vx")) / V_MAX
        vy_err = abs(e("vy") - g("vy")) / V_MAX
        vz_err = abs(e("vz") - g("vz")) / V_MAX
        yaw_err = abs(e("yaw_rate") - g("yaw_rate")) / R_MAX
        pitch_err = abs(e("pitch_rate") - g("pitch_rate")) / R_MAX
        roll_err = abs(e("roll_rate") - g("roll_rate")) / R_MAX

        # Scores
        vx_score = -vx_err + bonus(vx_err)
        vy_score = -vy_err + bonus(vy_err)
        vz_score = -vz_err + bonus(vz_err)
        yaw_score = -yaw_err + bonus(yaw_err)
        pitch_score = -pitch_err + bonus(pitch_err)
        roll_score = -roll_err + bonus(roll_err)

        total = (vx_score + vy_score + vz_score + yaw_score + pitch_score + roll_score) * MULT
        total = np.clip(total, -1000, 1000)

        return {
            "total": total,
            "vx_score": vx_score,
            "vy_score": vy_score,
            "vz_score": vz_score,
            "roll_score": roll_score,
            "pitch_score": pitch_score,
            "yaw_score": yaw_score,
            "yaw_rate": e("yaw_rate"),
            "pitch_rate": e("pitch_rate"),
            "roll_rate": e("roll_rate")
        }






    def is_terminal(self, state):
        return False