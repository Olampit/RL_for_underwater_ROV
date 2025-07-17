# run_policy.py

import time
import numpy as np
import torch
from pymavlink import mavutil

from imu_reader import start_imu_listener, stop_event, imu_thread, ros_thread
from environment import ROVEnvironment
from rov_env_gym import ROVEnvGymWrapper
from dac.dac_agent import DeterministicGCAgent

from joystick_input import FakeJoystick


def wait_for_heartbeat(conn, timeout=30):
    print("[WAIT] Waiting for MAVLink heartbeat…")
    conn.wait_heartbeat(timeout=timeout)
    print(f"[INFO] Connected: system={conn.target_system}, component={conn.target_component}")


def make_env(connection, latest_imu):
    rov_env = ROVEnvironment(action_map=[], connection=connection, latest_imu=latest_imu)
    return ROVEnvGymWrapper(rov_env)


def run_policy(
    actor_path="policy_actor.pth",
    mavlink_endpoint="udp:127.0.0.1:14550",
    max_steps=1000,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    conn = mavutil.mavlink_connection(mavlink_endpoint)
    wait_for_heartbeat(conn)

    joystick = FakeJoystick()

    latest_imu = {}
    start_imu_listener(conn, latest_imu, joystick)
    time.sleep(1)

    env = make_env(conn, latest_imu)
    obs = env.reset(conn)

    state_dim = obs.shape[1]
    action_dim = env.action_space.shape[0]
    sequence_dim = env.history_length

    agent = DeterministicGCAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        use_writer=False
    )
    agent.load_actor(actor_path)

    print("[INFO] Running policy…")

    try:
        for step in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action = agent.select_action(obs_tensor)

            next_obs, reward_components, done, _, _ = env.step(action)

            print(f"[STEP {step}] Reward: {reward_components['total']:.3f}")

            obs = next_obs

            if done:
                print("[DONE] Episode ended.")
                break

    finally:
        print("[CLEANUP] Stopping threads and motors.")
        stop_event.set()
        if imu_thread:
            imu_thread.join()
        if ros_thread:
            ros_thread.join()

        try:
            env.rov.stop_motors(conn)
        except Exception as e:
            print(f"[CLEANUP] Error stopping motors: {e}")

        try:
            env.close()
        except Exception as e:
            print(f"[CLEANUP] Error closing env: {e}")


if __name__ == "__main__":
    run_policy()
