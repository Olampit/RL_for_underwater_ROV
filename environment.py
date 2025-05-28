# environment.py
import aiohttp
import numpy as np
import math

class ROVEnvironment:
    def __init__(self, action_map, api_url="http://localhost:8080"):
        self.action_map = action_map
        self.api_url = api_url
        self.session = aiohttp.ClientSession()

    async def get_state(self):
        async with self.session.get(f"{self.api_url}/state") as resp:
            data = await resp.json()
            print(f"[STATE] {data}")
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

        await self.session.post(f"{self.api_url}/apply_action", json=command)

    def compute_reward(self, state):
        imu = state.get("IMU_COMBINED", {})
        ahrs2 = state.get("AHRS2", {})
        vibration = state.get("VIBRATION", {})
        
        # WE WILL NEED THE DVL FOR THIS
        v_x = self.current_velocity_x if hasattr(self, "current_velocity_x") else 0.0
        v_target = 0.5  # desired forward speed (m/s), will have to be passed way after. 
        
        # Acceleration penalties
        acc_y = imu.get("acc_y", 0.0)
        acc_z = imu.get("acc_z", 0.0)
        
        # Orientation
        roll = ahrs2.get("roll", 0.0)
        pitch = ahrs2.get("pitch", 0.0)
        yaw = ahrs2.get("yaw", 0.0)

        # Vibration
        vib_x = vibration.get("vibration_x", 0.0)
        vib_y = vibration.get("vibration_y", 0.0)
        vib_z = vibration.get("vibration_z", 0.0)

        # --- Reward Components ---
        alpha = 10.0  # sharpness of reward curve for velocity match
        speed_reward = math.exp(-alpha * (v_x - v_target) ** 2)

        # Penalties
        acc_penalty = abs(acc_y) + abs(acc_z)
        orientation_penalty = abs(roll) + abs(pitch)
        vibration_penalty = vib_x + vib_y + vib_z

        # Weights
        w_acc = 0.05
        w_orientation = 0.01
        w_vibration = 0.001

        # Total reward
        reward = (
            + speed_reward
            - w_acc * acc_penalty
            - w_orientation * orientation_penalty
            - w_vibration * vibration_penalty
        )

        print(f"[REWARD] v_x={v_x:.2f}, speed_reward={speed_reward:.2f}, acc_penalty={acc_penalty:.2f}, "
            f"orientation_penalty={orientation_penalty:.2f}, vib_penalty={vibration_penalty:.2f} → reward={reward:.2f}")
        
        return reward



    def is_terminal(self, state):
        depth = state.get("depth", 0.0)
        desired_speed = state.get("desired_speed", 0.0)

        if depth < -0.5 or depth > 10.0:
            print("[TERMINAL] Unsafe depth limit reached.")
            return True

        if abs(depth - 1.0) < 0.05 and abs(desired_speed) < 0.05:
            print("[TERMINAL] Target depth reached and stable.")
            return True

        return False

    async def close(self):
        await self.session.close()
