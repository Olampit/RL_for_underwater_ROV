# run_policy.py

import torch
import numpy as np
import time
from pymavlink import mavutil

from imu_reader import start_imu_listener
from joystick_input import FakeJoystick
from environment import ROVEnvironment
from rov_env_gym import ROVEnvGymWrapper
from dac.dac_agent import DeterministicGCAgent


def wait_for_heartbeat(conn, timeout=30):
    print("[WAIT] Waiting for MAVLink heartbeat...")
    conn.wait_heartbeat(timeout=timeout)
    print(f"[INFO] Connected to system {conn.target_system}, component {conn.target_component}")


def run_policy(
    model_path="policy_actor_save.pth",
    mavlink_endpoint="udp:127.0.0.1:14550",
    max_steps=1000,
    sleep_interval=0.1
):
    # 1. Connect to vehicle
    conn = mavutil.mavlink_connection(mavlink_endpoint)
    wait_for_heartbeat(conn)

    # 2. Start the IMU + goal logger
    latest_imu = {}
    joystick = FakeJoystick(evaluation_mode=True)
    joystick.set_manual_goal({
        "vx": 0.4,
        "vy": 0.0,
        "vz": 0.0,
        "yaw_rate": 0.0,
        "pitch_rate": 0.0,
        "roll_rate": 0.0,
    })

    start_imu_listener(conn, latest_imu, joystick)

    # 3. Create env wrapper
    time.sleep(1)  # Give IMU some time to start
    env = ROVEnvGymWrapper(ROVEnvironment([], conn, latest_imu))

    # 4. Init agent + load trained policy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DeterministicGCAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        use_writer=False  # Disable TensorBoard logging
    )
    agent.actor.load_state_dict(torch.load(model_path, map_location=device))
    agent.actor.eval()

    # 5. Run policy
    obs = env.reset(conn)
    print("[RUN] Starting policy rollout...")



    SEQ_LEN = 5
    state_buffer = []
            
    for step in range(max_steps):
        current_state = env.rov.get_state()
        print(current_state)

        # Get action from policy (no exploration)
        with torch.no_grad():
            

            state_buffer.append(obs)
            if len(state_buffer) > SEQ_LEN:
                state_buffer.pop(0)

            if len(state_buffer) < SEQ_LEN:
                action = np.zeros(action_dim)  # wait for enough context
            else:
                state_seq = np.array(state_buffer).astype(np.float32)  # shape: (5, state_dim)
                state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(device)  # (1, 5, state_dim)
                action = agent.actor(state_tensor).cpu().numpy()[0]

        # Apply action
        obs, reward_components, done, _ = env.step(action, current_state)
        time.sleep(sleep_interval)
        
        print(reward_components["total"])

        if env.rov.is_terminal(current_state):
            print(f"[INFO] Termination condition reached at step {step}")
            break

    # 6. Stop
    env.rov.stop_motors(conn)
    print("[DONE] Policy execution finished.")


if __name__ == "__main__":
    run_policy()
