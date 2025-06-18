# run_policy.py
import torch
import numpy as np
import time
from pymavlink import mavutil
from imu_reader import start_imu_listener
from environment import ROVEnvironment
from rov_env_gym import ROVEnvGymWrapper
from sac.sac_agent import SACAgent

def wait_for_heartbeat(conn, timeout=30):
    print("[WAIT] Waiting for MAVLink heartbeat...")
    conn.wait_heartbeat(timeout=timeout)
    print(f"[INFO] Connected to system {conn.target_system}, component {conn.target_component}")

def run():
    conn = mavutil.mavlink_connection("udp:127.0.0.1:14550")
    wait_for_heartbeat(conn)

    latest_imu = {}
    start_imu_listener(conn, latest_imu)
    time.sleep(2)

    env = ROVEnvGymWrapper(ROVEnvironment([], conn, latest_imu))

    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    agent.actor.load_state_dict(torch.load("sac_actor.pth"))
    agent.actor.eval()

    obs = env.reset()
    for step in range(1000):  # One test episode
        action = agent.select_action(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        time.sleep(0.1)
        if done:
            break

    env.rov.stop_motors(conn)
    print("[DONE] Episode finished.")

if __name__ == "__main__":
    run()
