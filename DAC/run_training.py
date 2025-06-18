# run_training_gc.py

import time
import numpy as np
from pymavlink import mavutil
import torch
import matplotlib.pyplot as plt

from imu_reader import start_imu_listener
from environment import ROVEnvironment
from rov_env_gym import ROVEnvGymWrapper
from deterministic_gc_agent import DeterministicGCAgent

def wait_for_heartbeat(conn, timeout=30):
    print("[WAIT] Waiting for MAVLink heartbeat…")
    conn.wait_heartbeat(timeout=timeout)
    print(f"[INFO] Connected: system={conn.target_system}, component={conn.target_component}")

def make_env(connection, latest_imu):
    rov_env = ROVEnvironment(action_map=[], connection=connection, latest_imu=latest_imu)
    return ROVEnvGymWrapper(rov_env)

def train(
    episodes=500,
    max_steps=20,
    batch_size=128,
    start_steps=1000,
    gamma=0.99,
    learning_rate=3e-4,
    device=None,
    mavlink_endpoint="udp:127.0.0.1:14550",
    progress_callback=None,
    pause_flag=None
):
    conn = mavutil.mavlink_connection(mavlink_endpoint)
    wait_for_heartbeat(conn)
    latest_imu = {}
    start_imu_listener(conn, latest_imu)
    time.sleep(1)

    env = make_env(conn, latest_imu)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dim = env.observation_space.shape[0]
    goal_dim = state_dim
    action_dim = env.action_space.shape[0]

    agent = DeterministicGCAgent(state_dim, goal_dim, action_dim, device=device, gamma=gamma, lr=learning_rate)

    episode_rewards = []
    total_steps = 0
    training_ended = False

    for ep in range(1, episodes + 1):
        if pause_flag and pause_flag.is_set():
            print("[PAUSED] Waiting to resume...")
            while pause_flag.is_set():
                time.sleep(0.5)

        obs = env.reset(conn)
        goal_dict = env.rov.joystick.get_target()
        goal = env._state_to_obs(env.rov._goal_to_state(goal_dict))
        ep_reward = 0.0
        total_step_time = 0

        for step in range(max_steps):
            t0 = time.time()

            if total_steps < start_steps:
                action = np.random.uniform(-1, 1, size=action_dim)
            else:
                action = agent.select_action(obs, goal)

            current_state = env.rov.get_state()
            next_obs, reward_components, done, _ = env.step(action, current_state)
            reward = reward_components["total"]

            agent.store_transition(obs, goal, action, reward, next_obs, done)

            obs = next_obs
            ep_reward += reward
            total_steps += 1

            critic_loss, actor_loss = agent.update(batch_size=batch_size)
            total_step_time += time.time() - t0

        episode_rewards.append(ep_reward)
        env.rov.stop_motors(conn)

        if progress_callback and ep % 50 == 0:
            target = goal_dict
            obs = env._state_to_obs(current_state)
            q_val = agent.critic(
                torch.FloatTensor(obs).unsqueeze(0).to(device),
                torch.FloatTensor(goal).unsqueeze(0).to(device),
                torch.FloatTensor(action).unsqueeze(0).to(device)
            ).item()

            metrics = {
                "vx": float(current_state.get("vx_mean", 0.0)),
                "vx_target": float(target.get("vx", {}).get("mean", 0.0)),
                "velocity_score": reward_components["velocity_score"],
                "yaw_rate": reward_components["yaw_rate"],
                "pitch_rate": reward_components["pitch_rate"],
                "roll_rate": reward_components["roll_rate"],
                "bonus": reward_components["bonus"],
                "stability_score": reward_components["stability_score"],
                "angular_score": reward_components["angular_score"],
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "entropy": 0.0,
                "mean_step_time": total_step_time / max_steps,
                "mean_q_value": q_val
            }
            progress_callback(ep, episodes, float(ep_reward), metrics)

    plt.plot(episode_rewards)
    plt.title("GC-Deterministic Actor-Critic: Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("gc_ac_rewards.pdf")
    print("[DONE] Saved reward plot to gc_ac_rewards.pdf")
