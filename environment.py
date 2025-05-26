import aiohttp
import numpy as np
from pymavlink import mavutil

SERVO_MIN = 1100
SERVO_MAX = 1900
SERVO_IDLE = 1500

def input_to_pwm(value):
    if abs(value) < 0.05:
        return SERVO_IDLE
    pwm = SERVO_IDLE + (value * 400)
    return int(max(SERVO_MIN, min(SERVO_MAX, pwm)))

class ROVEnvironment:
    def __init__(self, action_map, api_url="http://localhost:8080"):
        self.action_map = action_map
        self.api_url = api_url
        self.session = aiohttp.ClientSession()

        print("Connecting to ROV via MAVLink...")
        self.connection = mavutil.mavlink_connection('udp:0.0.0.0:14550')
        print("Waiting for ROV heartbeat...")
        self.connection.wait_heartbeat()
        print(f"Connected to system {self.connection.target_system}, component {self.connection.target_component}")

    async def get_state(self):
        async with self.session.get(f"{self.api_url}/state") as resp:
            data = await resp.json()
            return data

    def state_to_index(self, state):
        depth = state.get("depth", 0.0)
        pressure = state.get("pressure_abs", 1013.25)
        depth_idx = int(depth // 0.5)
        pressure_idx = int((pressure - 1000) // 1)
        return (depth_idx, pressure_idx)

    async def apply_action(self, action_index):
        command = self.action_map[action_index]
        print(f"[ACTION] Sending: {command}")

        await self.session.post(f"{self.api_url}/update", json=command)

        # Send to real ROV motors via MAVLink
        for motor_key, value in command.items():
            motor_num = int(motor_key.replace("motor", ""))
            pwm = input_to_pwm(value)
            print(f"Sending PWM {pwm} to motor {motor_num}")
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                motor_num,
                pwm,
                0, 0, 0, 0, 0
            )

    def compute_reward(self, state):
        target_depth = 1.0
        depth = state.get("depth", 0.0)
        desired_speed = state.get("desired_speed", 0.0)

        tracking_error = abs(depth - target_depth)
        stability_penalty = abs(desired_speed) * 0.1  # discourage rapid speed

        reward = -tracking_error - stability_penalty
        print(f"[REWARD] -depth error: {tracking_error:.2f}, -speed penalty: {stability_penalty:.2f} → reward = {reward:.2f}")
        return reward

    def is_terminal(self, state):
        depth = state.get("depth", 0.0)
        desired_speed = state.get("desired_speed", 0.0)
        
        # Condition 1: Out of safe depth range
        if depth < -0.5 or depth > 10.0:  # too shallow or too deep
            print("[TERMINAL] Unsafe depth limit reached.")
            return True

        # Condition 2: Target reached
        if abs(depth - 1.0) < 0.05 and abs(desired_speed) < 0.05:
            print("[TERMINAL] Target depth reached and stable.")
            return True

        # Add more conditions as needed...

        return False


    async def close(self):
        await self.session.close()
