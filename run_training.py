#run_training.py
from q_agent import QLearningAgent
from environment import ROVEnvironment
from imu_reader import start_imu_listener
from pymavlink import mavutil
import pickle
import itertools
import random
import time
import matplotlib.pyplot as plt

def sample_action_space(num_actions=32):
    motor_levels = [-1.0, 0.0, 1.0]
    all_combos = list(itertools.product(motor_levels, repeat=8))
    sampled_combos = random.sample(all_combos, num_actions)

    actions = []
    for combo in sampled_combos:
        action = {f"motor{i+1}": combo[i] for i in range(8)}
        actions.append(action)
    return actions

def wait_for_heartbeat(conn, timeout=30):
    print("[WAIT] Waiting for MAVLink heartbeat...")
    conn.wait_heartbeat(timeout=timeout)
    print(f"[INFO] Connected to system {conn.target_system}, component {conn.target_component}")

def train():
    # Setup MAVLink connection
    connection = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    wait_for_heartbeat(connection)

    latest_imu = {}
    start_imu_listener(connection, latest_imu)

    action_map = sample_action_space(num_actions=6561)

    print("[INIT] Waiting 2 seconds for IMU data to populate...")
    time.sleep(2)

    env = ROVEnvironment(action_map=action_map, connection=connection, latest_imu=latest_imu)
    agent = QLearningAgent(action_size=len(action_map))

    print("[TRAIN] Starting training...")

    step_rewards = []
    episode_rewards = []

    # Setup matplotlib live plots
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.set_title("Step Rewards (all episodes)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Reward")

    ax2.set_title("Episode Rewards")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")

    step_counter = 0

    for episode in range(6000):
        print(f"\n[EPISODE {episode + 1}]")

        state = env.get_state()
        state_idx = env.state_to_index(state)
        
        episode_reward = 0

        for step in range(10):
            action_idx = agent.choose_action(state_idx)
            env.apply_action(action_idx)
            
            
            
            time.sleep(2)

            next_state = env.get_state()
            next_state_idx = env.state_to_index(next_state)

            reward = env.compute_reward(next_state)
            agent.learn(state_idx, action_idx, reward, next_state_idx)

            episode_reward += reward
            step_rewards.append(reward)
            
            print(f"[STEP {step:02d}] Reward: {reward:.2f} | Pitch: {next_state.get('pitch', 0.0):.2f} | Roll: {next_state.get('roll', 0.0):.2f}")

            # Live update plot
            ax1.plot(step_rewards, color='blue')
            ax1.relim()
            ax1.autoscale_view()

            plt.pause(0.0001)
            ax1.cla()
            ax1.set_title("Step Rewards (all episodes)")
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Reward")
            ax1.plot(step_rewards, color='blue')

            state_idx = next_state_idx

            if env.is_terminal(next_state):
                print("[INFO] Terminal state reached. Ending episode.")
                break

        episode_rewards.append(episode_reward)

        # Update episode plot
        ax2.plot(episode_rewards, color='green')
        ax2.relim()
        ax2.autoscale_view()

        ax2.cla()
        ax2.set_title("Episode Rewards")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Total Reward")
        ax2.plot(episode_rewards, color='green')

        plt.pause(0.001)

        print(f"[EP {episode}] Total Reward: {episode_reward:.2f}")

    # Save learned Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)
    print("[SAVE] Q-table saved to q_table.pkl")

    # Keep final plot on screen
    plt.savefig("current.pdf")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train()

