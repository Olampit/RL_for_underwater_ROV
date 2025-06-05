#environment.py

from pymavlink import mavutil
import numpy as np
import time
import subprocess

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
        self.target_depth = 10.0
        self.target_yaw = 0.0

        vel_x = state.get("vel_x", 0.0)
        vel_y = state.get("vel_y", 0.0)
        vel_z = state.get("vel_z", 0.0)

        z = state.get("pos_z", 0.0)
        depth_error = abs(z - self.target_depth)

        yaw = state.get("yaw", 0.0)
        yaw_error = abs(yaw - self.target_yaw)

        # --- Directional reward: is depth improving?
        prev_error = getattr(self, "prev_depth_error", None)
        if prev_error is not None:
            depth_delta = prev_error - depth_error
            depth_improvement_reward = max(depth_delta, 0) * 5.0
        else:
            depth_improvement_reward = 0.0
        self.prev_depth_error = depth_error

        # --- Position penalty (distance from initial position)
        current_pos = {
            "x": state.get("pos_x", 0.0),
            "y": state.get("pos_y", 0.0),
            "z": state.get("pos_z", 0.0)
}



        dx = abs(current_pos["x"]) 
        dy = abs(current_pos["y"] - 50)
        dz = abs(current_pos["z"] - 10)
        spatial_drift = (dy**2 + dz**2) ** 0.5
        spatial_penalty = - (spatial_drift * 0.1)
        spatial_bonus = (dx * 5)
        
        # --- Core reward components
        forward_reward = max(vel_x, 0) / 0.5 * 3.0
        lateral_penalty = - (abs(vel_y) / 0.3 * 0.5)
        vertical_penalty = - (abs(vel_z) / 0.3 * 0.5)

        capped_depth_error = min(depth_error, 5.0)
        depth_penalty = - ((capped_depth_error / 2.0) * 0.5)

        yaw_penalty = - yaw_error / 2

        base_reward = 0
        reward = (
            base_reward
            + forward_reward
            + lateral_penalty
            + vertical_penalty
            + depth_penalty
            + yaw_penalty
            + spatial_penalty
            + depth_improvement_reward
            + spatial_bonus
        )

        if depth_error < 0.5 and abs(yaw_error) < 10:
            reward += 0.0

        return reward





    def state_to_index(self, state):
        keys = ["pitch", "roll", "vibration", "vel_x", "vel_y", "vel_z", "pos_x"]
        return tuple(round(state.get(k, 0.0), 0) for k in keys)



    def is_terminal(self, state):
        return abs(state.get("pitch", 0.0)) > 1.0 or abs(state.get("roll", 0.0)) > 1.0
