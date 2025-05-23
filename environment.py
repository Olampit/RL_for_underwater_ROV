#the environment for the learning agent. This links the aiohttp server and the q-learning agent, currently. 

import aiohttp
import numpy as np

class ROVEnvironment:
    def __init__(self, action_map, api_url="http://localhost:8080"):
        self.action_map = action_map
        self.api_url = api_url


    async def get_state(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_url}/state") as resp:
                return await resp.json()  # doit être un dict avec .get("depth")


    def state_to_index(self, state):
        # à adapter selon tes états
        return int(state["depth"] // 0.5)  # ex: discretisation naïve

    async def apply_action(self, action_index):
        command = self.action_map[action_index]
        print(f"[ACTION] Sending: {command}")
        async with aiohttp.ClientSession() as session:
            await session.post("http://localhost:8080/update", json=command)

    def compute_reward(self, state):
        # par exemple : rester à 1m de profondeur
        target = 1.0
        error = abs(state["depth"] - target)
        return -error
