#run the policy on the rov
import asyncio
import pickle
from environment import ROVEnvironment

my_actions = [
    {f"motor{i}": 0.0 for i in range(1, 9)},

    {f"motor{i}": 0.2 for i in range(1, 9)},

    {f"motor{i}": 0.5 for i in range(1, 9)},

    {f"motor{i}": -0.2 for i in range(1, 9)},

    {f"motor{i}": -0.5 for i in range(1, 9)},
] 

async def run():
    env = ROVEnvironment(action_map=my_actions, api_url="http://localhost:311")
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

    state = await env.get_state()
    state_idx = env.state_to_index(state)

    for step in range(50):
        action = int(np.argmax(q_table[state_idx]))
        print(f"[STEP {step}] Taking action: {action}")
        await env.apply_action(action)
        await asyncio.sleep(0.5)

        next_state = await env.get_state()
        print(f"Depth: {next_state['depth']:.2f}")
        state_idx = env.state_to_index(next_state)

if __name__ == "__main__":
    asyncio.run(run())
