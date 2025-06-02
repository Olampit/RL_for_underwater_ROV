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
    print("[INFO] Heartbeat received.")

def train():
    # MAVLink setup
    connection = mavutil.mavlink_connection('udp:127.0.0.1:14550')  # adjust port if needed
    wait_for_heartbeat(connection)

    # Shared state
    latest_imu = {}
    start_imu_listener(connection, latest_imu)

    # Environment and agent
    action_map = sample_action_space(num_actions=6000)
    env = ROVEnvironment(action_map=action_map, latest_imu=latest_imu, connection=connection)
    agent = QLearningAgent(action_size=len(action_map))

    print("[TRAIN] Starting training...")

    for episode in range(100):
        print(f"\n[EPISODE {episode}]")

        state = env.get_state()
        state_idx = env.state_to_index(state)

        for step in range(50):
            action = agent.choose_action(state_idx)
            env.apply_action(action)

            time.sleep(3)  # wait for ROV to respond

            next_state = env.get_state()
            next_state_idx = env.state_to_index(next_state)

            reward = env.compute_reward(next_state)
            agent.learn(state_idx, action, reward, next_state_idx)

            print(f"[STEP {step}] depth={state.get('depth', 0.0):.2f}, reward={reward:.2f}")

            state_idx = next_state_idx

            if env.is_terminal(next_state):
                break

    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)
        print("[SAVE] Q-table saved to q_table.pkl")


if __name__ == "__main__":
    train()
