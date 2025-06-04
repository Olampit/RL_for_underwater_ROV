#environment.py

from pymavlink import mavutil
import numpy as np

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

        return state

    def compute_reward(self, state):
        vel_x = state.get("vel_x", 0.0)
        vel_y = state.get("vel_y", 0.0)
        vel_z = state.get("vel_z", 0.0)

        # Scale for small ROV speeds
        forward = vel_x * 5.0           # Boost the impact of forward velocity (assuming < 0.5 m/s)
        lateral = -abs(vel_y) * 3.0     # Strong penalty for sideways movement
        vertical = -abs(vel_z) * 2.0    # Moderate penalty for vertical drift

        reward = forward + lateral + vertical

        # Optional: Clip extreme values (in case of spikes)
        reward = max(-10.0, min(10.0, reward))
        return reward


    def state_to_index(self, state):
        return tuple(round(state.get(k, 0.0), 1) for k in ["pitch", "roll", "vibration", "vel_x", "vel_y", "vel_z"])


    def is_terminal(self, state):
        return abs(state.get("pitch", 0.0)) > 1.0 or abs(state.get("roll", 0.0)) > 1.0
