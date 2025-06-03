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
                i + 1,  # servo number (assuming servo 1–8 map to motor1–motor8)
                pwm,
                0, 0, 0, 0, 0
            )
        print(f"[ACTION] Sent: {action}")

    def get_state(self):
        imu = self.latest_imu  # shortcut
        state = {}

        # Get pitch, roll, yaw from AHRS2 if available
        if "AHRS2" in imu:
            state.update({
                "pitch": imu["AHRS2"].get("pitch", 0.0),
                "roll": imu["AHRS2"].get("roll", 0.0),
                "yaw": imu["AHRS2"].get("yaw", 0.0)
            })

        # Vibration as norm
        if "VIBRATION" in imu:
            v = imu["VIBRATION"]
            state["vibration"] = np.linalg.norm([v.get("vibration_x", 0.0),
                                                 v.get("vibration_y", 0.0),
                                                 v.get("vibration_z", 0.0)])

        # Accel norm from IMU_COMBINED if available
        if "IMU_COMBINED" in imu:
            acc = imu["IMU_COMBINED"]
            state["accel"] = np.linalg.norm([acc["acc_x"], acc["acc_y"], acc["acc_z"]])

        return state

    def compute_reward(self, state):
        reward = 0.0

        # Orientation stability
        pitch = state.get("pitch", 0.0)
        roll = state.get("roll", 0.0)
        reward -= abs(pitch) * 0.6
        reward -= abs(roll) * 0.6

        # Vibration penalty
        vibration = state.get("vibration", 0.0)
        reward -= vibration * 0.1

        # Smooth acceleration
        accel = state.get("accel", 0.0)
        reward += max(0.0, 1.0 - abs(accel - 9.8)) * 0.2  # reward for being close to gravity

        # Depth goal
        if "depth" in state:
            depth_error = abs(state["depth"] - 2.0)  # target depth = 2.0 meters
            reward -= depth_error * 0.5

        return reward


    def state_to_index(self, state):
        pitch = round(state.get("pitch", 0.0), 1)
        roll = round(state.get("roll", 0.0), 1)
        vib = round(state.get("vibration", 0.0), 1)
        return hash((pitch, roll, vib)) % 10000

    def is_terminal(self, state):
        return abs(state.get("pitch", 0.0)) > 1.0 or abs(state.get("roll", 0.0)) > 1.0

