# run_training_sac.py
"""Soft Actor‑Critic training loop for the 8‑motor ROV.

This script:
1. Opens a MAVLink UDP connection to the simulator/hardware.
2. Starts the IMU/Odometry listener threads (unchanged from imu_reader.py).
3. Wraps the existing ROVEnvironment in a Gym‑style interface (ROVEnvGymWrapper).
4. Trains a PyTorch SACAgent with continuous 8‑dimensional actions in [‑1, 1].
5. Saves the learned actor network and a PDF of reward curves.

Requirements (install if missing):
    pip install torch gym matplotlib numpy
"""

import time
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymavlink import mavutil

# Local imports
from imu_reader import start_imu_listener
from environment import ROVEnvironment
from rov_env_gym import ROVEnvGymWrapper
from sac.sac_agent import SACAgent


def wait_for_heartbeat(conn, timeout: int = 30):
    print("[WAIT] Waiting for MAVLink heartbeat…")
    conn.wait_heartbeat(timeout=timeout)
    print(f"[INFO] Connected: system={conn.target_system}, component={conn.target_component}")


def make_env(connection, latest_imu):
    """Instantiate low‑level ROVEnvironment and wrap it with Gym adapter."""
    # action_map unused by SAC path ‑ pass empty list
    rov_env = ROVEnvironment(action_map=[], connection=connection, latest_imu=latest_imu)
    return ROVEnvGymWrapper(rov_env)


def train():
    # --- 1. Connect to MAVLink ‑ adjust the endpoint if needed
    conn = mavutil.mavlink_connection("udp:127.0.0.1:14550")
    wait_for_heartbeat(conn)

    # --- 2. Shared IMU dictionary populated by background threads
    latest_imu = {}
    start_imu_listener(conn, latest_imu)

    # Let sensors accumulate a little data
    print("[INIT] Waiting 2 s for IMU/Odometry data …")
    time.sleep(2)

    # --- 3. Env + agent
    env = make_env(conn, latest_imu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device,
    )

    # --- 4. Hyper‑parameters
    EPISODES = 1000
    MAX_STEPS = 300          # per episode
    BATCH_SIZE = 256
    START_STEPS = 5_000      # purely random actions before learning
    UPDATE_EVERY = 1         # train every step after buffer warm‑up
    GAMMA = 0.99
    REWARD_SCALE = 1.0

    # Tracking
    episode_rewards = []

    total_steps = 0
    for ep in range(1, EPISODES + 1):
        obs = env.reset()
        ep_reward = 0.0
        
        for step in range(1, MAX_STEPS + 1):
            # --- Select action
            if total_steps < START_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            # --- Execute
            next_obs, reward, done, _ = env.step(action)

            # --- Store in replay buffer
            agent.replay_buffer.push(obs, action, reward * REWARD_SCALE, next_obs, done)

            obs = next_obs
            ep_reward += reward
            total_steps += 1

            # --- Update agent
            if total_steps >= START_STEPS and total_steps % UPDATE_EVERY == 0:
                agent.update(batch_size=BATCH_SIZE)

            if done:
                print(f"[EP {ep:03d}] Done at step {step} | Reward={ep_reward:.2f}")
                break

        episode_rewards.append(ep_reward)

        # Safety: stop all motors between episodes
        env.rov.stop_motors(conn)
        time.sleep(0.5)

        # Simple progress print
        if ep % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"[INFO] Episode {ep}, 10‑ep avg reward = {avg:.2f}")

    # --- 5. Save results
    torch.save(agent.actor.state_dict(), "sac_actor.pth")
    print("[SAVE] Actor network saved to sac_actor.pth")

    # Plot rewards
    plt.figure(figsize=(10, 4))
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(episode_rewards)
    plt.tight_layout()
    plt.savefig("sac_training_rewards.pdf")
    print("[DONE] Training curve saved to sac_training_rewards.pdf")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    train()
