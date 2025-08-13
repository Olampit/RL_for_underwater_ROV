#run_policy.py
import time
import numpy as np
from pymavlink import mavutil
import torch

from imu_reader import start_imu_listener, stop_event, imu_thread, ros_thread
from environment import ROVEnvironment
from rov_env_gym import ROVEnvGymWrapper
from dac.dac_agent import DeterministicGCAgent
from joystick_input import FakeJoystick

import os
import traceback


def wait_for_heartbeat(conn, timeout=30):
    print("[WAIT] Waiting for MAVLink heartbeat…")
    conn.wait_heartbeat(timeout=timeout)
    print(f"[INFO] Connected: system={conn.target_system}, component={conn.target_component}")

def load_actor_from_checkpoint(agent, checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "net" in ckpt and "actor" in ckpt["net"]:
        agent.actor.load_state_dict(ckpt["net"]["actor"])
        print(f"[LOAD] Actor loaded from {checkpoint_path}")
    else:
        raise KeyError(f"No actor weights found in checkpoint: {checkpoint_path}")

def make_env(connection):
    rov_env = ROVEnvironment(action_map=[], connection=connection)
    return ROVEnvGymWrapper(rov_env)

def set_servo_function(servo_number, connection, value=0):
    """
    Set the function of a specific servo channel on the ROV.
    This is often used to disable/enable motors during setup or reset.
    """
    
    param_name = f"SERVO{servo_number}_FUNCTION"
    param_name = param_name.encode("utf-8")

    connection.mav.param_set_send(
        connection.target_system,
        connection.target_component,
        param_name,
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    
    print(f"{param_name.decode()} set to {value}")


def run_policy_with_goals(
    model_path="checkpoints/latest.pt",
    goals=None,
    steps_per_goal=200,
    device=None,
    mavlink_endpoint="udp:127.0.0.1:14550"
):
    """
    Runs a trained deterministic policy through a list of preset goals.
    Each goal is held for `steps_per_goal` steps.
    """
    if goals is None:
        goals = [
            {"vx": 0.5, "vy": 0.0, "vz": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            {"vx": -0.5, "vy": 0.0, "vz": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            {"vx": 0.0, "vy": 0.5, "vz": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            {"vx": 0.0, "vy": -0.5, "vz": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        ]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Connect
    conn = mavutil.mavlink_connection(mavlink_endpoint)
    wait_for_heartbeat(conn)
    
    # Enable motors
    for i in range(1, 9):
        set_servo_function(i, conn, 0)


    # Joystick with manual goal injection
    joystick = FakeJoystick()

    # Start IMU listener
    start_imu_listener(conn, joystick)
    time.sleep(1)

    # Environment
    env = make_env(conn)
    obs = env.reset(conn)
    state_dim = obs.shape[1]
    action_dim = env.action_space.shape[0]

    # Agent
    agent = DeterministicGCAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        use_writer=False
    )
    load_actor_from_checkpoint(model_path)

    try:
        for goal_idx, goal in enumerate(goals):
            joystick.goal = goal  # Directly set goal
            print(f"[GOAL {goal_idx+1}/{len(goals)}] {goal}")
            ep_reward = 0.0
            for step in range(steps_per_goal):
                action = agent.select_action(obs)
                next_obs, reward_components, done, _, _ = env.step(action)
                ep_reward += reward_components.get("total", 0.0)
                obs = next_obs
                if done:
                    break

            print(f"    Reward for goal {goal_idx+1}: {ep_reward:.3f}")

    except Exception as e:
        print(f"[ERROR] Exception in policy run: {e}")
        traceback.print_exc()
    finally:
        print("[CLEANUP] Stopping imu listener threads...")
        stop_event.set()
        if imu_thread:
            imu_thread.join()
        if ros_thread:
            ros_thread.join()

        print("[CLEANUP] Stopping motors and closing environment.")
        try:
            env.rov.stop_motors(conn)
        except Exception as e:
            print(f"[CLEANUP] Error stopping motors: {e}")
        try:
            env.close()
        except Exception as e:
            print(f"[CLEANUP] Error closing env: {e}")
        print("[DONE] Policy loop exited.")



if __name__ == "__main__":
    run_policy_with_goals(
        model_path="policy_actor.pth",
        goals=[
            {"vx": 0.5, "vy": 0.0, "vz": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            {"vx": -0.5, "vy": 0.0, "vz": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        ],
        steps_per_goal=200,
        device="cuda"  # or "cpu"
    )
