#run_training.py
import asyncio
from q_agent import QLearningAgent
from environment import ROVEnvironment
import pickle
import itertools
import random
import aiohttp
from aiohttp import web

import aiohttp
import asyncio

async def wait_for_server(api_url, timeout=30):
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{api_url}/state") as resp:
                    if resp.status == 200:
                        try:
                            data = await resp.json()
                            # Optionnel : on peut vérifier des clés minimales
                            if isinstance(data, dict):  # JSON bien formé
                                print("[INFO] Server ready.")
                                return
                        except Exception as e:
                            print(f"[WARN] Invalid JSON: {e}")
        except aiohttp.ClientError as e:
            print(f"[WAIT] Server not ready yet: {e}")
        await asyncio.sleep(1)
    raise RuntimeError("Server did not become ready in time.")

def sample_action_space(num_actions=32):
    motor_levels = [-1.0, 0.0, 1.0]
    all_combos = list(itertools.product(motor_levels, repeat=8))
    sampled_combos = random.sample(all_combos, num_actions)

    actions = []
    for combo in sampled_combos:
        action = {f"motor{i+1}": combo[i] for i in range(8)}
        actions.append(action)
    return actions

my_actions = sample_action_space(num_actions=6000)

async def train():
    api_url = "http://localhost:8080"
    await wait_for_server(api_url)

    env = ROVEnvironment(action_map=my_actions, api_url=api_url)  
    agent = QLearningAgent(action_size=len(my_actions))

    state = await env.get_state()

    for episode in range(100):
        state = await env.get_state()
        state_idx = env.state_to_index(state)

        for step in range(50):
            action = agent.choose_action(state_idx)
            await env.apply_action(action)

            await asyncio.sleep(0.5) ###################################################################""

            next_state = await env.get_state()
            next_state_idx = env.state_to_index(next_state)

            reward = env.compute_reward(next_state)
            agent.learn(state_idx, action, reward, next_state_idx)

            print(f"Step {step}, State: {state['depth']:.2f}, Reward: {reward:.2f}")

            state_idx = next_state_idx

    await env.close()

    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)

    
if __name__ == "__main__":
    asyncio.run(train())
