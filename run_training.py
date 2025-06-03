#run_training.py
from q_agent import QLearningAgent
from environment import ROVEnvironment
from imu_reader import start_imu_listener
from pymavlink import mavutil
import pickle
import itertools
import random
import time

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
    connection = mavutil.mavlink_connection('udp:127.0.0.1:14550')  # Adjust for your setup
    wait_for_heartbeat(connection)

    # Initialize shared IMU dict and start background listener
    latest_imu = {}
    start_imu_listener(connection, latest_imu)

    # Sample action space
    action_map = sample_action_space(num_actions=6561)

    # Wait briefly to accumulate IMU data
    print("[INIT] Waiting 2 seconds for IMU data to populate...")
    time.sleep(2)

    # Initialize environment and agent
    env = ROVEnvironment(action_map=action_map, connection=connection, latest_imu=latest_imu)
    agent = QLearningAgent(action_size=len(action_map))

    print("[TRAIN] Starting training...")

    for episode in range(100):
        print(f"\n[EPISODE {episode + 1}]")

        state = env.get_state()
        state_idx = env.state_to_index(state)

        for step in range(50):
            action_idx = agent.choose_action(state_idx)
            env.apply_action(action_idx)

            time.sleep(0.5)  # Wait for ROV to respond

            next_state = env.get_state()
            next_state_idx = env.state_to_index(next_state)

            reward = env.compute_reward(next_state)
            agent.learn(state_idx, action_idx, reward, next_state_idx)

            print(f"[STEP {step:02d}] Reward: {reward:.2f} | Pitch: {next_state.get('pitch', 0.0):.2f} | Roll: {next_state.get('roll', 0.0):.2f}")

            state_idx = next_state_idx

            if env.is_terminal(next_state):
                print("[INFO] Terminal state reached. Ending episode.")
                break

    # Save learned Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)
    print("[SAVE] Q-table saved to q_table.pkl")

if __name__ == "__main__":
    train()

