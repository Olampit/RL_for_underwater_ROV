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
        target_depth = 1.0
        depth = state.get("depth", 0.0)
        desired_speed = state.get("desired_speed", 0.0)

        tracking_error = abs(depth - target_depth)
        stability_penalty = abs(desired_speed) * 0.1

        reward = -tracking_error - stability_penalty
        print(f"[REWARD] -depth error: {tracking_error:.2f}, -speed penalty: {stability_penalty:.2f} → reward = {reward:.2f}")
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
