# run_policy.py

import torch
import numpy as np
import time
from pymavlink import mavutil

from imu_reader import start_imu_listener
from environment import ROVEnvironment
from rov_env_gym import ROVEnvGymWrapper
from dac.dac_agent import DeterministicGCAgent

def wait_for_heartbeat(conn, timeout=30):
    print("[WAIT] Waiting for MAVLink heartbeat...")
    conn.wait_heartbeat(timeout=timeout)
    print(f"[INFO] Connected to system {conn.target_system}, component {conn.target_component}")

def run_policy(
    model_path="policy_actor.pth",
    mavlink_endpoint="udp:127.0.0.1:14550",
    max_steps=1000,
    sleep_interval=0.1
):
    conn = mavutil.mavlink_connection(mavlink_endpoint)
    wait_for_heartbeat(conn)

    latest_imu = {}
    start_imu_listener(conn, latest_imu)
    time.sleep(1)

    env = ROVEnvGymWrapper(ROVEnvironment([], conn, latest_imu))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DeterministicGCAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    agent.actor.load_state_dict(torch.load(model_path, map_location=device))
    agent.actor.eval()

    obs = env.reset(conn)

    for step in range(max_steps):
        current_state = env.rov.get_state()
        action = agent.select_action(obs, deterministic=True)
        obs, reward_components, done, _ = env.step(action, current_state)
        time.sleep(sleep_interval)

        if env.rov.is_terminal(current_state):
            print(f"[INFO] Termination condition reached at step {step}")
            break

    env.rov.stop_motors(conn)
    print("[DONE] Policy execution finished.")

if __name__ == "__main__":
    run_policy()
