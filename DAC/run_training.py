# run_training_gc.py

import time
import numpy as np
from pymavlink import mavutil
import torch
import matplotlib.pyplot as plt

from imu_reader import start_imu_listener, stop_event, imu_thread, ros_thread
from environment import ROVEnvironment
from rov_env_gym import ROVEnvGymWrapper
from dac.dac_agent import DeterministicGCAgent

import threading
import traceback
import sys
from tkinter import messagebox

import requests


def wait_for_heartbeat(conn, timeout=30):
    print("[WAIT] Waiting for MAVLink heartbeat…")
    conn.wait_heartbeat(timeout=timeout)
    print(f"[INFO] Connected: system={conn.target_system}, component={conn.target_component}")

def make_env(connection, latest_imu):
    rov_env = ROVEnvironment(action_map=[], connection=connection, latest_imu=latest_imu)
    return ROVEnvGymWrapper(rov_env)

def safe_scalar(x):
    import numpy as np
    import torch

    if isinstance(x, (np.ndarray, list, tuple)):
        if len(x) == 1:
            return float(x[0])
        elif len(x) > 1:
            return float(x[0])
        else:
            return 0.0
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.item())
        elif x.numel() > 1:
            return float(x.flatten()[0].item())
        else:
            return 0.0
    else:
        return float(x)
    
def set_servo_function(servo_number, connection, value=0):
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

def train(
    episodes=500,
    max_steps=20,
    batch_size=1024,
    start_steps=5000,
    gamma=0.99,
    learning_rate_start=5e-2,
    learning_rate_end=1e-4,
    device=None,
    mavlink_endpoint="udp:127.0.0.1:14550",
    progress_callback=None,
    pause_flag=None,
    shutdown_flag=None
):
    conn = mavutil.mavlink_connection(mavlink_endpoint)
    wait_for_heartbeat(conn)
    latest_imu = {}
    start_imu_listener(conn, latest_imu)
    time.sleep(1)
    
    
    update_every = 1

    env = make_env(conn, latest_imu)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dim = env.observation_space.shape[0]
    goal_dim = state_dim
    action_dim = env.action_space.shape[0]

    agent = DeterministicGCAgent(state_dim, goal_dim, action_dim, device=device, gamma=gamma, lr=learning_rate_start, lr_end=learning_rate_end)

  
    
    episode_rewards = []
    total_steps = 0
    
    restart_countdown = 1000
    url = "http://localhost/ardupilot-manager/v1.0/restart"

    for i in range(1, 9):
        set_servo_function(i, conn, 0)
        
    try:
        for ep in range(5, episodes + 6):
            if shutdown_flag and shutdown_flag.is_set():
                print("[STOP] Shutdown flag detected. Ending training...")
                break

            if pause_flag and pause_flag.is_set():
                print("[PAUSED] Waiting to resume...")
                while pause_flag.is_set():
                    if shutdown_flag and shutdown_flag.is_set():
                        print("[STOP] Shutdown during pause. Exiting.")
                        return
                    time.sleep(0.5)

    
            
            if restart_countdown == 0:
                print("resetting firmware")
                response = requests.post(url)
                time.sleep(120)
                for i in range(1, 9):
                    set_servo_function(i, conn, 0)
                restart_countdown = 1000
            else : 
                restart_countdown -= 1 
            
            
            if ep%5 == 0:
                obs = env.reset(conn)
            goal_dict = env.rov.joystick.get_target()
            goal = env._state_to_obs(env.rov._goal_to_state(goal_dict))
            ep_reward = 0.0
            total_step_time = 0

            for step in range(max_steps):
                if shutdown_flag and shutdown_flag.is_set():
                    print("[STOP] Shutdown during episode.")
                    return

                t0 = time.time()

                if total_steps < start_steps:
                    action = agent.sample_random_structured()
                    
                else:
                    action = agent.select_action(obs, goal)

                current_state = env.rov.get_state()
                next_obs, reward_components, done, _ = env.step(action, current_state)
                reward = reward_components["total"]

                agent.replay_buffer.push(obs, goal, action, reward, next_obs, done)

                obs = next_obs
                ep_reward += reward
                total_steps += 1

                if total_steps % update_every == 0 : 
                    update_info = agent.update(batch_size=batch_size, total_step=total_steps)
                    critic_loss = update_info.get("critic_loss", 0.0)
                    actor_loss = update_info.get("actor_loss", 0.0)

                agent.lr_step(total_steps, lr_start=learning_rate_start, lr_end=learning_rate_end)
                
                total_step_time += time.time() - t0
                


            episode_rewards.append(ep_reward)
            env.rov.stop_motors(conn)

            
            if progress_callback:
                target = goal_dict
                obs = env._state_to_obs(current_state)
                obs = np.asarray(obs).astype(np.float32).flatten()
                goal = np.asarray(goal).astype(np.float32).flatten()
                action = np.asarray(action).astype(np.float32).flatten()

                q_val = agent.critic(
                    torch.FloatTensor(obs).unsqueeze(0).to(device),
                    torch.FloatTensor(goal).unsqueeze(0).to(device),
                    torch.FloatTensor(action).unsqueeze(0).to(device)
                ).item()
                
                

                metrics = {
                    "vx": safe_scalar(current_state.get("vx", 0.0)),
                    "vx_target": safe_scalar(target.get("vx", {}).get("mean", 0.0)),
                    "vy": safe_scalar(current_state.get("vy", 0.0)),
                    "vz": safe_scalar(current_state.get("vz", 0.0)),
                    
                    "yaw_rate": safe_scalar(reward_components.get("yaw_rate", 0.0)),
                    "pitch_rate": safe_scalar(reward_components.get("pitch_rate", 0.0)),
                    "roll_rate": safe_scalar(reward_components.get("roll_rate", 0.0)),
                    
                    "vx_score": safe_scalar(reward_components.get("vx_score", 0.0)),
                    "vy_score": safe_scalar(reward_components.get("vy_score", 0.0)),
                    "vz_score": safe_scalar(reward_components.get("vz_score", 0.0)),
                    "roll_score": safe_scalar(reward_components.get("roll_score", 0.0)),
                    "pitch_score": safe_scalar(reward_components.get("pitch_score", 0.0)),
                    "yaw_score": safe_scalar(reward_components.get("yaw_score", 0.0)),
                    


                    "critic_loss": safe_scalar(critic_loss),
                    "actor_loss": safe_scalar(actor_loss),
                    "mean_step_time": safe_scalar(total_step_time) / max_steps,
                    "mean_q_value": safe_scalar(q_val),
                    
                    
                    "td_mean": safe_scalar(update_info.get("td_mean", 0.0)),
                    "td_max": safe_scalar(update_info.get("td_max", 0.0)),
                    "td_min": safe_scalar(update_info.get("td_min", 0.0)),
                    "actor_grad_norm": safe_scalar(update_info.get("actor_grad_norm", 0.0)),
                    "critic_grad_norm": safe_scalar(update_info.get("critic_grad_norm", 0.0)),
                    "actor_weight_norm": safe_scalar(update_info.get("actor_weight_norm", 0.0)),
                    "critic_weight_norm": safe_scalar(update_info.get("critic_weight_norm", 0.0)),
                    "learning_rate":safe_scalar(update_info.get("learning_rate", 0.0))
                    
                    
                }

                progress_callback(ep, episodes, float(ep_reward), metrics)
                
        torch.save(agent.actor.state_dict(), "policy_actor.pth")
        torch.save(agent.critic.state_dict(), "policy_critic.pth")

    except Exception as e:
        print(f"[ERROR] Exception in training: {e}")
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
        print("[DONE] Training loop exited.")
    
    


def run_training(self, agent_type, config):
    try:
        if agent_type == "sac":
            train(**config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    except Exception as e:
        error_details = "".join(traceback.format_exception(*sys.exc_info()))
        self.log("Error occurred:\n" + error_details)
        messagebox.showerror("Training Error", f"An error occurred:\n\n{str(e)}\n\nCheck log for full traceback.")
    finally:
        self.notify_training_finished()
