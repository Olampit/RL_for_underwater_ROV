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
        print(f"[DEBUG] apply_action called with index {action_idx}")
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
                "pitch_speed" :a.get("pitchspeed",0.0),
                "yaw" : a.get("yaw", 0.0),
                "yaw_speed" : a.get("yawspeed", 0.0),
                "roll" : a.get("roll", 0.0),
                "roll_speed" : a.get("rollspeed",0.0)
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
        # Random small variation in spawn position 
        px = round(random.uniform(-1.0, 1.0), 2)
        py = round(random.uniform(4.0, 6), 2)
        pz = round(random.uniform(9.5, 10.5), 2)
        
        

        # Random orientation
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

        print(f"[RESET] Respawning ROV at pos=({px:.2f}, {py:.2f}, {pz:.2f}) with random orientation. : x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}, w: {qw:.6f}")
        subprocess.run(cmd)

        
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

    
    def total_distance(self, state):
        x = state.get("pos_x", 0.0)
        return x



    def compute_reward(self, state):
        self.goal = self.joystick.get_target()

        # Raw velocity
        vx = state.get("vel_x", 0.0)
        vy = state.get("vel_y", 0.0)
        vz = state.get("vel_z", 0.0)

        # Target velocity
        vx_target = self.goal.get("vx", 0.0)
        vy_target = self.goal.get("vy", 0.0)
        vz_target = self.goal.get("vz", 0.0)

        # Error (L1 distance)
        vel_error = abs(vx - vx_target) + abs(vy - vy_target) + abs(vz - vz_target)

        # Angular stability 
        yaw_rate = abs(state.get("yaw_speed", 0.0))
        pitch_rate = abs(state.get("pitch_speed", 0.0))
        roll_rate = abs(state.get("roll_speed", 0.0))
        angular_energy = yaw_rate + pitch_rate + roll_rate

        # Reward shaping
        progress_reward = -vel_error                # lower error = better
        stability_penalty = -angular_energy         # lower spin = better
        bonus = 5.0 if vel_error < 0.02 and angular_energy < 0.02 else 0.0

        # Total reward
        total = 2.0 * progress_reward + 1.0 * stability_penalty + bonus  # weights here

        return {
            "total": total,
            "progress_reward": progress_reward,
            "stability": stability_penalty,
            "bonus": bonus,
            "vel_error" : vel_error,
            "yaw_rate" : yaw_rate,
            "pitch_rate" : pitch_rate,
            "roll_rate" : roll_rate
        }








    def state_to_index(self, state):
        keys = ["pitch_speed", "roll_speed", "yaw_speed", "vel_x", "vel_y", "vel_z"]
        return tuple(round(state.get(k, 0.0), 2) for k in keys)




    def is_terminal(self, state):
        
        return False

