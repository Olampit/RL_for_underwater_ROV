#environment.py

from pymavlink import mavutil
import numpy as np
import time
import subprocess
from joystick_input import FakeJoystick

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

        # Orientation
        if "AHRS2" in imu:
            state.update({
                "pitch": imu["AHRS2"].get("pitch", 0.0),
                "roll": imu["AHRS2"].get("roll", 0.0),
                "yaw": imu["AHRS2"].get("yaw", 0.0)
            })

        # Vibration
        if "VIBRATION" in imu:
            v = imu["VIBRATION"]
            state["vibration"] = np.linalg.norm([
                v.get("vibration_x", 0.0),
                v.get("vibration_y", 0.0),
                v.get("vibration_z", 0.0)
            ])

        # Acceleration
        if "IMU_COMBINED" in imu:
            acc = imu["IMU_COMBINED"]
            state["accel"] = np.linalg.norm([
                acc.get("acc_x", 0.0),
                acc.get("acc_y", 0.0),
                acc.get("acc_z", 0.0)
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

    def reset(self):
        # appeler le script reset
        subprocess.run(["./reset_rov.sh"])
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
        # Update target velocity from joystick
        self.target_velocity = self.joystick.get_target()

        # Actual state
        vx = state.get("vel_x", 0.0)
        vy = state.get("vel_y", 0.0)
        vz = state.get("vel_z", 0.0)
        z = state.get("pos_z", 0.0)
        yaw_rate = state.get("yaw_rate", 0.0)  # add this to your state if not yet
        yaw = state.get("yaw", 0.0)

        # Target state
        vx_tgt = self.target_velocity["vx"]
        vy_tgt = self.target_velocity["vy"]
        vz_tgt = self.target_velocity["vz"]
        depth_tgt = self.target_velocity["depth"]
        yaw_rate_tgt = self.target_velocity["yaw_rate"]

        # Errors
        vel_error = ((vx - vx_tgt) ** 2 + (vy - vy_tgt) ** 2 + (vz - vz_tgt) ** 2)
        depth_error = abs(z - depth_tgt)
        yaw_rate_error = abs(yaw_rate - yaw_rate_tgt)

        # Reward terms
        forward_reward = max(vx, 0) * 5.0                   # strongly encourage forward motion
        vel_penalty = - vel_error * 3.0                     # punish deviation from target velocity
        depth_penalty = - min(depth_error, 5.0) * 2.0       # punish deviation from target depth
        yaw_spin_penalty = - abs(yaw_rate) * 2.0            # discourage spinning
        yaw_error_penalty = - abs(yaw) * 0.1                # penalize yaw drift (optional)

        # Bonus if agent is close to ideal behavior
        bonus = 0.0
        if vel_error < 0.01 and depth_error < 0.2 and abs(yaw_rate) < 0.1:
            bonus += 2.0

        # Total reward
        reward = (
            forward_reward
            + vel_penalty
            + depth_penalty
            + yaw_spin_penalty
            + yaw_error_penalty
            + bonus
        )

        return reward








    def state_to_index(self, state):
        keys = ["pitch", "roll", "vibration", "vel_x", "vel_y", "vel_z", "pos_x"]
        return tuple(round(state.get(k, 0.0), 0) for k in keys)



    def is_terminal(self, state):
        return abs(state.get("pitch", 0.0)) > 1.0 or abs(state.get("roll", 0.0)) > 1.0
