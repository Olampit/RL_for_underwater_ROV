# run_training.py
from q_agent import QLearningAgent
from environment import ROVEnvironment
from imu_reader import start_imu_listener
from pymavlink import mavutil
import pickle
import itertools
import random
import time

import matplotlib
matplotlib.use('Agg')  # Headless backend for safe plotting
import matplotlib.pyplot as plt


def sample_action_space(num_actions=32):
    motor_levels = [-1.0, 0.0, 1.0]
    all_combos = list(itertools.product(motor_levels, repeat=8))
    sampled_combos = random.sample(all_combos, num_actions)
    actions = [{f"motor{i+1}": combo[i] for i in range(8)} for combo in sampled_combos]
    return actions

def wait_for_heartbeat(conn, timeout=30):
    print("[WAIT] Waiting for MAVLink heartbeat...")
    conn.wait_heartbeat(timeout=timeout)
    print(f"[INFO] Connected to system {conn.target_system}, component {conn.target_component}")

def train():
    connection = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    wait_for_heartbeat(connection)

    latest_imu = {}
    start_imu_listener(connection, latest_imu)

    action_map = sample_action_space(num_actions=6561)

    print("[INIT] Waiting 2 seconds for IMU data to populate...")
    time.sleep(2)

    env = ROVEnvironment(action_map=action_map, connection=connection, latest_imu=latest_imu)
    agent = QLearningAgent(action_size=len(action_map))

    print("[RESET] Stopping and resetting ROV")
    env.stop_motors(connection)
    state = env.reset()
    print("[TRAIN] Starting training...")

    step_rewards = []
    episode_rewards = []
    episode_distances = []

    for episode in range(6000):
        env.stop_motors(connection)
        state = env.reset()
        state_idx = env.state_to_index(state)

        print(f"\n[EPISODE {episode + 1}]")
        episode_reward = 0

        for step in range(100):
            action_idx = agent.choose_action(state_idx)
            env.apply_action(action_idx)
            time.sleep(0.1)

            next_state = env.get_state()
            next_state_idx = env.state_to_index(next_state)

            reward = env.compute_reward(next_state)
            agent.learn(state_idx, action_idx, reward, next_state_idx)

            episode_reward += reward
            step_rewards.append(reward)

            print(f"[STEP {step:02d}] Reward: {reward:.2f} | Pitch: {next_state.get('pitch', 0.0):.2f} | Roll: {next_state.get('roll', 0.0):.2f}")

            state_idx = next_state_idx

            if env.is_terminal(next_state):
                print("[INFO] Terminal state reached. Ending episode.")
                break

        episode_rewards.append(episode_reward)
        final_state = env.get_state()
        dx = env.total_distance(final_state)
        episode_distances.append(dx)

        print(f"[EP {episode}] Total Reward: {episode_reward:.2f}")

    # Save Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)
    print("[SAVE] Q-table saved to q_table.pkl")

    # Plot results once at the end
    print("[PLOT] Saving final training plots...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Step Rewards (Last 500)")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.plot(step_rewards[-500:], color='blue')

    plt.subplot(1, 2, 2)
    plt.title("Distance from Spawn (X)")
    plt.xlabel("Episode")
    plt.ylabel("Distance (m)")
    plt.plot(episode_distances, color='green')

    plt.tight_layout()
    plt.savefig("training_summary.pdf")
    print("[DONE] Plot saved to training_summary.pdf")

if __name__ == "__main__":
    train()
