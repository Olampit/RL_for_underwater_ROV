#here, we can just execute this file to start the training. 
import asyncio
from q_agent import QLearningAgent
from environment import ROVEnvironment
import pickle
import itertools
import random

def sample_action_space(num_actions=32):
    motor_levels = [-1.0, 0.0, 1.0]
    all_combos = list(itertools.product(motor_levels, repeat=8))
    sampled_combos = random.sample(all_combos, num_actions)

    actions = []
    for combo in sampled_combos:
        action = {f"motor{i+1}": combo[i] for i in range(8)}
        actions.append(action)
    return actions

my_actions = sample_action_space(num_actions=256)

async def train():
    env = ROVEnvironment(action_map=my_actions, api_url="http://localhost:8080")
    agent = QLearningAgent(action_size=len(my_actions))

    for episode in range(100):
        state = await env.get_state()
        state_idx = env.state_to_index(state)

        for step in range(50):
            action = agent.choose_action(state_idx)
            await env.apply_action(action)

            await asyncio.sleep(5)  # laisser le ROV bouger un peu

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
