#environment.py

from pymavlink import mavutil
import numpy as np
import time
import subprocess
from joystick_input import FakeJoystick
import math
import random

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
        print(f"[ACTION] Sent: {action}")

    def get_state(self):
        imu = self.latest_imu
        state = {}


        # ATTITUDE
        if "ATTITUDE" in imu:
            a = imu["ATTITUDE"]
            state.update({
                "pitch" : a.get("pitch", 0.0),
                "pitchspeed" :a.get("pitchspeed",0.0),
                "yaw" : a.get("yaw", 0.0),
                "yawspeed" : a.get("yawspeed", 0.0),
                "roll" : a.get("roll", 0.0),
                "rollspeed" : a.get("rollspeed",0.0)
            })


        # Vibration
        if "VIBRATION" in imu:
            v = imu["VIBRATION"]
            state["vibration"] = np.linalg.norm([
                v.get("vibration_x", 0.0),
                v.get("vibration_y", 0.0),
                v.get("vibration_z", 0.0)
            ])


        # Velocity from Odometry
        if "ODOMETRY" in imu and "velocity" in imu["ODOMETRY"]:
            vel = imu["ODOMETRY"]["velocity"]
            state.update({
                "vel_x": vel.get("x", 0.0),
                "vel_y": vel.get("y", 0.0),
                "vel_z": vel.get("z", 0.0)
            })
            
        if "ODOMETRY" in imu and "position" in imu["ODOMETRY"]:
            pos = imu["ODOMETRY"]["position"]
            state.update({
                "pos_x": pos.get("x", 0.0),
                "pos_y": pos.get("y", 0.0),
                "pos_z": pos.get("z", 0.0)
            })


        MAX_POS = 100.0   # assume ROV won't be outside [-100, 100]
        MAX_VEL = 1.0     # assume velocities stay under +/- 1 m/s
        MAX_ANGLE = np.pi  # roll/pitch/yaw in radians
        MAX_YAW_speed = 2.0  # rad/s is a safe ceiling for spin rate



        state.update({
            "norm_pos_x": np.clip(state.get("pos_x", 0.0) / MAX_POS, -1.0, 1.0),
            "norm_pos_y": np.clip(state.get("pos_y", 0.0) / MAX_POS, -1.0, 1.0),
            "norm_pos_z": np.clip(state.get("pos_z", 0.0) / MAX_POS, -1.0, 1.0),

            "norm_vel_x": np.clip(state.get("vel_x", 0.0) / MAX_VEL, -1.0, 1.0),
            "norm_vel_y": np.clip(state.get("vel_y", 0.0) / MAX_VEL, -1.0, 1.0),
            "norm_vel_z": np.clip(state.get("vel_z", 0.0) / MAX_VEL, -1.0, 1.0),

            "norm_pitch": np.clip(state.get("pitch", 0.0) / MAX_ANGLE, -1.0, 1.0),
            "norm_roll": np.clip(state.get("roll", 0.0) / MAX_ANGLE, -1.0, 1.0),
            "norm_yaw": np.clip(state.get("yaw", 0.0) / MAX_ANGLE, -1.0, 1.0),


            "norm_yaw_speed": np.clip(state.get("yaw_speed", 0.0) / MAX_YAW_speed, -1.0, 1.0),
            "norm_roll_speed": np.clip(state.get("roll_speed", 0.0) / MAX_YAW_speed, -1.0, 1.0),
            "norm_pitch_speed": np.clip(state.get("pitch_speed", 0.0) / MAX_YAW_speed, -1.0, 1.0)

        })

        return state

    def random_orientation_quat(max_angle_deg=15):
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
        # Random small variation in spawn position 
        px = round(random.uniform(-1.0, 1.0), 2)
        py = round(random.uniform(49.0, 51.0), 2)
        pz = round(random.uniform(9.5, 10.5), 2)

        # Random orientation
        quat = random_orientation_quat(max_angle_deg=15)
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

        print(f"[RESET] Respawning ROV at pos=({px:.2f}, {py:.2f}, {pz:.2f}) with random orientation. : x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}, w: {qw:.6f}")
        subprocess.run(cmd)

        self.no_progress_steps = 0
        self.prev_distance = float("inf")

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

    
    def total_distance(self, state):
        x = state.get("pos_x", 0.0)
        return abs(x)


    def compute_reward(self, state):
        self.goal = self.joystick.get_target()

        # Actual velocities
        vx = state.get("vel_x", 0.0)
        vy = state.get("vel_y", 0.0)
        vz = state.get("vel_z", 0.0)

        # Angular velocities
        yaw_rate = state.get("yawspeed", 0.0)
        pitch_rate = state.get("pitchspeed", 0.0)
        roll_rate = state.get("rollspeed", 0.0)

        # Target velocity
        vx_target = self.goal.get("vx", 0.0)
        vy_target = self.goal.get("vy", 0.0)
        vz_target = self.goal.get("vz", 0.0)

        # --- Velocity error
        err_vx = abs(vx - vx_target)
        err_vy = abs(vy - vy_target)
        err_vz = abs(vz - vz_target)

        # === Dynamic blending weight ===
        # More instability = more weight on stabilization
        angular_energy = abs(yaw_rate) + abs(pitch_rate) + abs(roll_rate)
        stab_weight = min(1.0, angular_energy / 1.5)  # blend up to full at ~1.5 rad/s total

        # === Velocity reward (encourage forward motion)
        vel_reward = (
            max(1.0 - err_vx / 0.3, 0.0) * 5.0 +
            max(1.0 - err_vy / 0.3, 0.0) * 2.0 +
            max(1.0 - err_vz / 0.3, 0.0) * 2.0
        )

        # === Stability reward (encourage zero angular velocity)
        stab_reward = - min(abs(yaw_rate), 2.0) * 1.0 - min(abs(pitch_rate), 2.0) * 1.0 - min(abs(roll_rate), 2.0) * 1.0

        # === Combined
        reward = (
            (1.0 - stab_weight) * vel_reward +
            stab_weight * stab_reward
        )

        # === Bonus if stable and accurate
        if err_vx < 0.05 and err_vy < 0.05 and err_vz < 0.05 and angular_energy < 0.1:
            reward += 3.0

        return reward







    def state_to_index(self, state):
        keys = ["norm_pitch_speed", "norm_roll_speed", "norm_yaw_speed", "norm_vel_x", "norm_vel_y", "norm_vel_z"]
        return tuple(round(state.get(k, 0.0), 2) for k in keys)




    def is_terminal(self, state):
        
        if abs(state.get("pitch", 0.0)) > 1.0 or abs(state.get("roll", 0.0)) > 1.0:
            return True
        
        return False

