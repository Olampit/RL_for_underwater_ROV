#here, we can just execute this file to start the training. 
import asyncio
from q_agent import QLearningAgent
from environment import ROVEnvironment
import pickle


my_actions = [
    {f"motor{i}": 0.0 for i in range(1, 9)},

    {f"motor{i}": 0.2 for i in range(1, 9)},

    {f"motor{i}": 0.5 for i in range(1, 9)},

    {f"motor{i}": -0.2 for i in range(1, 9)},

    {f"motor{i}": -0.5 for i in range(1, 9)},
]

async def train():
    env = ROVEnvironment(action_map=my_actions, api_url="http://localhost:8080")
    agent = QLearningAgent(action_size=len(my_actions))

    for episode in range(100):
        state = await env.get_state()
        state_idx = env.state_to_index(state)

        for step in range(50):
            action = agent.choose_action(state_idx)
            await env.apply_action(action)

            await asyncio.sleep(1)  # laisser le ROV bouger un peu

            next_state = await env.get_state()
            next_state_idx = env.state_to_index(next_state)

            reward = env.compute_reward(next_state)
            agent.learn(state_idx, action, reward, next_state_idx)

            print(f"Step {step}, State: {state['depth']:.2f}, Reward: {reward:.2f}")

            state_idx = next_state_idx  # passer à l’état suivant


    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)
    
    
if __name__ == "__main__":
    asyncio.run(train())
