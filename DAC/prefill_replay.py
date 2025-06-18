# prefill_replay.py
from sac.sac_agent import SACAgent
from run_training import make_env, wait_for_heartbeat
from environment import ROVEnvironment
from imu_reader import start_imu_listener
from pymavlink import mavutil
import time
import torch

def prefill_replay_buffer(env, agent, steps=50000, reward_scale=1.0):
    """
    Populates the replay buffer by running the environment with random actions.

    Parameters:
        env (ROVEnvGymWrapper): Gym environment for the ROV.
        agent (SACAgent): Agent with a replay buffer to populate.
        steps (int): Number of steps to fill.
        reward_scale (float): Multiplier applied to each reward before storing.

    Called in:
        Standalone, should be deprecated since there is a copy in run_training.py.
    """
    obs = env.reset()
    for _ in range(steps):
        action = env.action_space.sample()
        next_obs, _, done, _ = env.step(action)
        current_state = env.rov.get_state()
        reward_components = env.rov.compute_reward(current_state)
        reward = reward_components["total"]
        agent.replay_buffer.push(obs, action, reward * reward_scale, next_obs, done)
        obs = next_obs
        if done:
            obs = env.reset()
        time.sleep(0.01)  # Optional: slow down random exploration
    print("[DONE] Replay buffer pre-filled.")

if __name__ == "__main__":
    conn = mavutil.mavlink_connection("udp:127.0.0.1:14550")
    wait_for_heartbeat(conn)
    latest_imu = {}
    start_imu_listener(conn, latest_imu)
    time.sleep(2)

    env = make_env(conn, latest_imu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
    )
    prefill_replay_buffer(env, agent, steps=50000, reward_scale=1.0)
    agent.replay_buffer.save("replay_buffer.pkl")
