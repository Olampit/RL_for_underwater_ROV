# environment.py
import aiohttp
import numpy as np


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

        # Acceleration
        acc_x = imu.get("acc_x", 0.0)
        acc_y = imu.get("acc_y", 0.0)
        acc_z = imu.get("acc_z", 0.0)

        # Gyro
        gyro_x = imu.get("gyro_x", 0.0)
        gyro_y = imu.get("gyro_y", 0.0)
        gyro_z = imu.get("gyro_z", 0.0)

        # Orientation
        roll = ahrs2.get("roll", 0.0)
        pitch = ahrs2.get("pitch", 0.0)
        yaw = ahrs2.get("yaw", 0.0)

        # Vibration
        vib_x = vibration.get("vibration_x", 0.0)
        vib_y = vibration.get("vibration_y", 0.0)
        vib_z = vibration.get("vibration_z", 0.0)

        # Weights
        weight_x_accel = 0.02
        weight_yz_accel = 0.05
        weight_gyro = 0.02
        weight_orientation = 0.01
        weight_vibration = 0.001

        reward = (
            + weight_x_accel * acc_x
            - weight_yz_accel * (abs(acc_y) + abs(acc_z))
            - weight_gyro * (abs(gyro_x) + abs(gyro_y) + abs(gyro_z))
            - weight_orientation * (abs(roll) + abs(pitch) + abs(yaw))
            - weight_vibration * (vib_x + vib_y + vib_z)
        )

        print(f"[REWARD] acc_x={acc_x:.2f}, penalty_yz_accel={abs(acc_y)+abs(acc_z):.2f}, "
            f"gyro_penalty={abs(gyro_x)+abs(gyro_y)+abs(gyro_z):.2f}, "
            f"orientation_penalty={abs(roll)+abs(pitch)+abs(yaw):.2f}, "
            f"vibration_penalty={vib_x+vib_y+vib_z:.2f} → reward={reward:.2f}")

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
