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

from joystick_input import FakeJoystick

import os



def wait_for_heartbeat(conn, timeout=30):
    """Block until a MAVLink heartbeat is received, indicating the ROV is ready."""
    print("[WAIT] Waiting for MAVLink heartbeat…")
    conn.wait_heartbeat(timeout=timeout)
    print(f"[INFO] Connected: system={conn.target_system}, component={conn.target_component}")

def make_env(connection):
    """Create the Gym-style ROV environment with wrapping."""
    rov_env = ROVEnvironment(action_map=[], connection=connection)
    return ROVEnvGymWrapper(rov_env)

def safe_scalar(x):
    """
    Convert tensors, arrays or scalars to a safe float.
    Useful for logging with TensorBoard or UI, avoids shape/format issues.
    """
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

def train(
    episodes=500,
    max_steps=20,
    batch_size=32,
    update_every = 30,
    start_steps=5000,  #exploration, make it bigger...
    gamma=0.99,
    learning_rate_start=5e-2,
    learning_rate_end=1e-4,
    device=None,
    mavlink_endpoint="udp:127.0.0.1:14550",
    progress_callback=None,
    pause_flag=None,
    shutdown_flag=None
):
    
    """
    Main training loop:
    - Connects to the ROV over MAVLink
    - Initializes agent and environment
    - Runs episodes, collects experience
    - Performs training updates on the actor-critic
    """
    
    
    os.makedirs("checkpoints", exist_ok=True)
    
    exploration_steps = start_steps * max_steps                   # Phase 1: e.g., 2000 steps
    actor_learning_steps = 200_000  * max_steps                    # Phase 2: after exploration, before noisy validation

    def get_training_phase(step):
        if step < exploration_steps:
            return "exploration"
        elif step < actor_learning_steps:
            return "actor_learning"
        else:
            return "noisy_learning"

    
    # Connect to ROV and wait for MAVLink readiness
    conn = mavutil.mavlink_connection(mavlink_endpoint)
    wait_for_heartbeat(conn)
    
    
    # Create joystick goal generator
    joystick = FakeJoystick()
    
    
    # Start listener for IMU and sensor data collection
    start_imu_listener(conn, joystick)
    time.sleep(1) # Give sensor thread time to warm up
    
    
    # Frequency of updates in environment steps (can be tuned in the function call)
    update_every = update_every  #! update needs to be large (~1000 - 5000 maybe)
    
    
    # Set device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu" 

    
    # Initialize environment and get dimensionsq
    env = make_env(conn)
    obs = env.reset(conn)
    state_dim = obs.shape[1] 
    action_dim = env.action_space.shape[0]


    # Set shared/global dimensions
    state_dimension = state_dim
    sequence_dimension = env.history_length
    
    # Instantiate the agent
    agent = DeterministicGCAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        use_writer=False
    )
    
    
    episode_rewards = []    # Logs episode return for debugging
    total_steps = 1         # Total env steps (global counter)
    
    
    # --- Firmware restart configuration (blueos restart) ---
    restart_countdown = 1000
    url = "http://localhost/ardupilot-manager/v1.0/restart"


    # enable motors initially to permit movement
    for i in range(1, 5): # set only the first 8 motors currently
        set_servo_function(i, conn, 0)
        
    
    # Initialize placeholders for losses and logging
    critic_loss = 0.0
    actor_loss = 0.0
    
    update_info = {
        "critic_loss": 0.0,
        "actor_loss": 0.0,
        "td_mean": 0.0, 
        "td_max": 0.0,
        "td_min": 0.0,
        "actor_grad_norm": 0.0,
        "critic_grad_norm": 0.0,
        "actor_weight_norm": 0.0,
        "critic_weight_norm": 0.0,
        "learning_rate": 0.0
    }
    
        
    # ------------------- Training Loop -------------------
    try:
        for ep in range(5, episodes + 6): # Start at episode 5 for safety
            
            
            if ep % 10_000 == 0:  # Save every 10_000 episodes
                torch.save(agent.actor.state_dict(), f"checkpoints/actor_ep{ep}.pth")
                torch.save(agent.critic.state_dict(), f"checkpoints/critic_ep{ep}.pth")
            
            phase = get_training_phase(total_steps)
            
            # Exit if shutdown triggered
            if shutdown_flag and shutdown_flag.is_set():
                print("[STOP] Shutdown flag detected. Ending training...")
                break


            # Pause loop if pause flag is active
            if pause_flag and pause_flag.is_set():
                print("[PAUSED] Waiting to resume...")
                while pause_flag.is_set():
                    if shutdown_flag and shutdown_flag.is_set():
                        print("[STOP] Shutdown during pause. Exiting.")
                        return
                    time.sleep(0.5)

    
            # restart firmware to avoid blueos crashes
            if restart_countdown == 0:
                print("resetting firmware")
                response = requests.post(url)
                time.sleep(120) # Give blueos time to reboot
                for i in range(1, 5): #! only 8 motors here too
                    set_servo_function(i, conn, 0)
                restart_countdown = 1000
            else : 
                restart_countdown -= 1 
            
            
            # Periodically reset the environment
            if ep%5 == 0:
                obs = env.reset(conn)
                
                
                
            # Update joystick target 
            if phase in ["exploration"]and ep%update_every == 0:
                joystick.switch_goal_randomly()
                
            if phase in ["actor_learning"] and ep%update_every == 0 :
                joystick.switch_goal_randomly()
                
            



            ep_reward = 0.0
            total_step_time = 0
            
            
            # === Epsilon-Greedy Exploration ===
            exploration_bool = total_steps < start_steps # Use random actions before buffer fills


            
                
            for step in range(max_steps):
                if shutdown_flag and shutdown_flag.is_set():
                    print("[STOP] Shutdown during episode.")
                    return

                t0 = time.time()
                
                
                if phase == "exploration":
                    action = agent.sample_random_structured(action_dim)  # Random action
                elif phase == "actor_learning":
                    if np.random.randint(0,100) < 90:
                        action = agent.select_action(obs)          # Deterministic actor (no noise)
                    else : 
                        action = agent.sample_random_structured(action_dim)
                else:  # "noisy_learning"
                    action = agent.select_action(obs, noise_std=0.05)  # Add noise for robustness



                # Step
                next_obs, reward_components, done, _, current_state = env.step(action, exploration_bool)

                reward = reward_components["total"]
                

                # Convert observations and action to appropriate format
                obs = np.asarray(obs, dtype=np.float32).reshape(sequence_dimension, state_dimension)
                next_obs = np.asarray(next_obs, dtype=np.float32).reshape(sequence_dimension, state_dimension)
                action = np.asarray(action, dtype=np.float32).flatten()
                
                
                
                done = (step == max_steps - 1) or done # since we reset right after, we should NOT predict on this particular step
                # Store transition in replay buffer
                agent.replay_buffer.push(obs, action, reward, next_obs, done)


                # Update episode state
                obs = next_obs
                ep_reward += reward
                total_steps += 1
            
            
                # === Perform periodic training update ===
                if total_steps % update_every == 0:
                    update_info = agent.update(batch_size=batch_size, total_step=total_steps)

                    critic_loss = update_info.get("critic_loss", 0.0)
                    actor_loss = update_info.get("actor_loss", 0.0)


                # === Learning rate decay over time ===
                agent.lr_step(total_steps, lr_start=learning_rate_start, lr_end=learning_rate_end)
                
                
                # Measure time taken for this step (for diagnostics)
                total_step_time += time.time() - t0
                

            # === End of episode ===
            episode_rewards.append(ep_reward)
            
            
            #update goal from third part here so we can have the reward_components 
            if phase in ["noisy_learning"]:
                joystick.update_success_tracking()
            
            
            # Immediately stop all motors to prevent drift or accidents
            env.rov.stop_motors(conn)

            
            # === report metrics to GUI ===
            if progress_callback:
                # Recompute state and Q-value for latest transition
                obs = env._state_to_obs()
                obs = np.asarray(obs).astype(np.float32).flatten()
                action = np.asarray(action).astype(np.float32).flatten()

                
                
                c_goal = joystick.get_target()

                # === Metrics Dictionary ===
                metrics = {
                    "reward_total": safe_scalar(reward_components.get("total", 0.0)),

                    # --- Actual velocities/orientations from ROV ---
                    "vx": safe_scalar(reward_components.get("vx", 0.0)),
                    "vy": safe_scalar(reward_components.get("vy", 0.0)),
                    "vz": safe_scalar(reward_components.get("vz", 0.0)),
                    "yaw": safe_scalar(reward_components.get("yaw", 0.0)),
                    "pitch": safe_scalar(reward_components.get("pitch", 0.0)),
                    "roll": safe_scalar(reward_components.get("roll", 0.0)),

                    # --- Goal targets from reward output ---
                    "goal_vx": safe_scalar(reward_components.get("goal_vx", 0.0)),
                    "goal_vy": safe_scalar(reward_components.get("goal_vy", 0.0)),
                    "goal_vz": safe_scalar(reward_components.get("goal_vz", 0.0)),
                    "goal_yaw": safe_scalar(reward_components.get("goal_yaw", 0.0)),
                    "goal_pitch": safe_scalar(reward_components.get("goal_pitch", 0.0)),
                    "goal_roll": safe_scalar(reward_components.get("goal_roll", 0.0)),

                    # --- Error signals ---
                    "vx_error": safe_scalar(reward_components.get("vx_error", 0.0)),
                    "vy_error": safe_scalar(reward_components.get("vy_error", 0.0)),
                    "vz_error": safe_scalar(reward_components.get("vz_error", 0.0)),
                    "yaw_error": safe_scalar(reward_components.get("yaw_error", 0.0)),
                    "pitch_error": safe_scalar(reward_components.get("pitch_error", 0.0)),
                    "roll_error": safe_scalar(reward_components.get("roll_error", 0.0)),
                    "std_penalty": safe_scalar(reward_components.get("std_penalty", 0.0)),

                    # --- Individual reward scores ---
                    "vx_score": safe_scalar(reward_components.get("velocity_alignment", 0.0)),
                    "vy_score": safe_scalar(reward_components.get("orientation_alignment", 0.0)),
                    "vz_score": safe_scalar(reward_components.get("vz_score", 0.0)),
                    "yaw_score": safe_scalar(reward_components.get("yaw_score", 0.0)),
                    "pitch_score": safe_scalar(reward_components.get("pitch_score", 0.0)),
                    "roll_score": safe_scalar(reward_components.get("roll_score", 0.0)),
                    "direction_bonus": safe_scalar(reward_components.get("direction_bonus", 0.0)),

                    # --- Learning rate ---
                    "learning_rate": safe_scalar(update_info.get("learning_rate", 0.0)),

                    # --- TD error & gradient stats ---
                    "td_mean": safe_scalar(update_info.get("td_mean", 0.0)),
                    "td_max": safe_scalar(update_info.get("td_max", 0.0)),
                    "td_min": safe_scalar(update_info.get("td_min", 0.0)),
                    "actor_grad_norm": safe_scalar(update_info.get("actor_grad_norm", 0.0)),
                    "critic_grad_norm": safe_scalar(update_info.get("critic_grad_norm", 0.0)),
                    "actor_weight_norm": safe_scalar(update_info.get("actor_weight_norm", 0.0)),
                    "critic_weight_norm": safe_scalar(update_info.get("critic_weight_norm", 0.0)),

                    # --- Losses ---
                    "critic_loss": safe_scalar(critic_loss),
                    "actor_loss": safe_scalar(actor_loss),
                    "mean_step_time": safe_scalar(total_step_time) / max_steps,
                    "mean_q_value": safe_scalar(0),
                }


                # Push all metrics to UI
                progress_callback(ep, episodes, float(ep_reward), metrics)
        
        
        # === End of training: Save models ===
        torch.save(agent.actor.state_dict(), "policy_actor.pth")
        torch.save(agent.critic.state_dict(), "policy_critic.pth")


    # --- Error handling ---
    except Exception as e:
        print(f"[ERROR] Exception in training: {e}")
        traceback.print_exc()


    # === Cleanup (always run) ===
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
    
    

# === Wrapper for UI (Tkinter) ===
def run_training(self, agent_type, config):
    """
    Wrapper to launch training based on selected agent type.
    Catches and reports any error to GUI and log.
    """
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



