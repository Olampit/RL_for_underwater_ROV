# run_training_sac.py
"""Soft Actor-Critic training loop for the 8-motor ROV (GUI-ready).

This version exposes **train(**kwargs) so it can be called from a tkinter GUI or
from the CLI. All hyper-parameters are configurable through keyword arguments
and sensible defaults are provided.

Key kwargs (all optional):
    episodes            (int)   - number of training episodes (default 1000)
    max_steps           (int)   - max steps per episode (default 300)
    batch_size          (int)   - mini-batch size for the SAC update (256)
    start_steps         (int)   - purely random steps before updates (5000)
    update_every        (int)   - update frequency in env steps (1)
    reward_scale        (float) - multiplier applied before storing in buffer
    learning_rate       (float) - Adam LR for both actor & critic (3e-4)
    gamma               (float) - discount factor (0.99)
    tau                 (float) - target-net soft update (0.005)
    mavlink_endpoint    (str)   - MAVLink URL ("udp:127.0.0.1:14550")
    device              (str)   - "cpu", "cuda" or None to auto-select
    progress_callback   (callable | None)
                                A function called after every episode with
                                signature cb(episode_idx:int, total:int, reward:float)

The script can still be run directly:
    python run_training_sac.py --episodes 300 --max_steps 200
"""

from __future__ import annotations

import cProfile


import argparse
import time
from typing import Callable, Optional, Dict, Any

import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymavlink import mavutil
import os
import pickle
import threading


# Local imports
from imu_reader import start_imu_listener
from environment import ROVEnvironment
from rov_env_gym import ROVEnvGymWrapper
from sac.sac_agent import SACAgent

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

SPEED_UP=5

def wait_for_heartbeat(conn, timeout: int = 30):
    print("[WAIT] Waiting for MAVLink heartbeat…")
    conn.wait_heartbeat(timeout=timeout)
    print(f"[INFO] Connected: system={conn.target_system}, component={conn.target_component}")


def make_env(connection, latest_imu):
    """Instantiate low-level ROVEnvironment and wrap it with Gym adapter."""
    rov_env = ROVEnvironment(action_map=[], connection=connection, latest_imu=latest_imu)
    return ROVEnvGymWrapper(rov_env)


def save_checkpoint(agent, total_steps, episode_rewards, filename="sac_checkpoint.pt"):
    print(f"[SAVE] Saving checkpoint at step {total_steps}...")
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'actor_opt': agent.actor_opt.state_dict(),
        'critic_opt': agent.critic_opt.state_dict(),
        'replay_buffer': agent.replay_buffer,
        'step': total_steps,
        'rewards': episode_rewards,
    }, filename)
    
    
def load_checkpoint(agent, filename="save/sac_checkpoint.pt"):
    if not os.path.exists(filename):
        return 0, []

    print("[LOAD] Loading checkpoint…")
    checkpoint = torch.load(filename, map_location=agent.device)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    agent.actor_opt.load_state_dict(checkpoint['actor_opt'])
    agent.critic_opt.load_state_dict(checkpoint['critic_opt'])
    agent.replay_buffer = checkpoint['replay_buffer']
    return checkpoint['step'], checkpoint['rewards']


def prefill_replay_buffer(env, agent, conn, steps=50000, reward_scale=0.01):
    obs = env.reset(conn)
    for _ in range(steps):
        action = env.action_space.sample()
        current_state = env.rov.get_state()
        next_obs, _, done, _ = env.step(action, current_state)
        reward_components = env.rov.compute_reward(current_state)
        reward = reward_components["total"]
        agent.replay_buffer.push(obs, action, reward * reward_scale, next_obs, done)
        obs = next_obs
        if done:
            obs = env.reset()




# -----------------------------------------------------------------------------
# Main train() callable
# -----------------------------------------------------------------------------

def train(
    *,
    episodes: int = 5000,
    max_steps: int = 10,
    batch_size: int = 50,#TODO FIX FUCKING BATCH SIZE
    start_steps: int = 0, #!
    update_every: int = 50,
    reward_scale: float = 1,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    mavlink_endpoint: str = "udp:127.0.0.1:14550",
    device: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    resume: bool = False,
    checkpoint_every: int = 1000,
    pause_flag: Optional[threading.Event] = None,
    restart_flag: Optional[threading.Event] = None,
) -> Dict[str, Any]:


    critic_losses = []
    actor_losses = []
    entropies = []
    critic_loss = 0.0
    actor_loss = 0.0
    entropy = 0.0



    conn = mavutil.mavlink_connection(mavlink_endpoint)
    wait_for_heartbeat(conn)
    latest_imu: Dict[str, Any] = {}
    start_imu_listener(conn, latest_imu)
    print("[INIT] Waiting 2 s for IMU/Odometry data …")
    time.sleep(2)

    env = make_env(conn, latest_imu)
    env.episode_states = []

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device,
        gamma=gamma,
        tau=tau,
        alpha=0.2,
    )
    


    for opt in (agent.actor_opt, agent.critic_opt):
        for param_group in opt.param_groups:
            param_group["lr"] = learning_rate

    buffer_path = "replay_buffer.pkl"
    prefill_steps = 250

    if os.path.exists(buffer_path):
        agent.replay_buffer.load(buffer_path)
    else:
        print(f"[INFO] Prefilling replay buffer with {prefill_steps} random steps...")
        prefill_replay_buffer(env, agent, conn, steps=prefill_steps, reward_scale=reward_scale)
        agent.replay_buffer.save(buffer_path)

    if resume:
        total_steps, episode_rewards = load_checkpoint(agent)
        start_ep = len(episode_rewards) + 1
    else:
        episode_rewards = []
        total_steps = 0
        start_ep = 1

    ep = start_ep
    
    while ep <= episodes:
        if restart_flag and restart_flag.is_set():
            print("[INFO] Restart flag set. Resetting episode counter.")
            episode_rewards = []
            total_steps = 0
            ep = 0 #!1 ?
            restart_flag.clear()

        if pause_flag and pause_flag.is_set():
            print("[PAUSED] Waiting to resume...")
            while pause_flag.is_set():
                time.sleep(0.5)

        obs = env.reset(conn)
        ep_reward = 0.0
        
        
        
        step_time = 0 
        total_step_time = 0
        for step in range(1, max_steps + 1):
            
            step_time = time.time()
            
            
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)


            current_state = env.rov.get_state()

            next_obs, _, done, _ = env.step(action, current_state)
            
            x = max((0.025/SPEED_UP)-(time.time()-step_time), 0)
            time.sleep(x)####################################################
            
            total_step_time += time.time()-step_time

            
            reward_components = env.rov.compute_reward(current_state)
            reward = reward_components["total"]

            agent.replay_buffer.push(obs, action, reward * reward_scale, next_obs, done)

            obs = next_obs
            ep_reward += reward
            total_steps += 1
            
            if total_steps >= start_steps and total_steps % update_every == 0:
                critic_loss, actor_loss, entropy = agent.update(batch_size=batch_size)
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
                entropies.append(entropy)


            if done:
                break

            if total_steps > 0 and total_steps % checkpoint_every == 0:
                save_checkpoint(agent, total_steps, episode_rewards)
            
                

        
        
        env.rov.stop_motors(conn)


        episode_rewards.append(ep_reward)
        env.episode_states.append(current_state)


        if progress_callback is not None and step % 50 == 0:
            target = env.rov.joystick.get_target()
            
            metrics = {
                "vx": float(current_state.get("vel_x", 0.0)),
                "vx_target": float(target.get("vx", {}).get("mean", 0.0)),
                "progress_reward": reward_components["progress_reward"],
                "yaw_rate": reward_components["yaw_rate"],
                "pitch_rate": reward_components["pitch_rate"],
                "roll_rate": reward_components["roll_rate"],
                "bonus": reward_components["bonus"],
                "stability": reward_components["stability"],
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "entropy": entropy * 10,
                "mean_step_time": (total_step_time/max_steps),
            }
            progress_callback(ep, episodes, float(ep_reward), metrics)


        if ep % 10000 == 0:
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'actor_opt': agent.actor_opt.state_dict(),
                'critic_opt': agent.critic_opt.state_dict(),
                'replay_buffer': agent.replay_buffer,
                'step': total_steps,
                'rewards': episode_rewards,
            }, f"save/sac_actor_ep{ep:04d}_step{total_steps}.pth")

        ep += 1

    torch.save(agent.actor.state_dict(), "sac_actor.pth")
    print("[SAVE] Actor network saved to sac_actor.pth")

    plt.figure(figsize=(10, 4))
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(episode_rewards)
    plt.tight_layout()
    plt.savefig("sac_training_rewards.pdf")
    print("[DONE] Training curve saved to sac_training_rewards.pdf")

    return {
        "episode_rewards": episode_rewards,
        "total_steps": total_steps,
        "model_path": "sac_actor.pth",
        "plot_path": "sac_training_rewards.pdf",
    }



# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC for ROV control")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--mavlink", type=str, default="udp:127.0.0.1:14550")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

        #
        # train(
        # episodes=args.episodes,
        # max_steps=args.max_steps,
        # learning_rate=args.learning_rate,
        # mavlink_endpoint=args.mavlink,
        # )

    # Profile the call to train with your args
    cProfile.runctx(
        'train(episodes=args.episodes, max_steps=args.max_steps, learning_rate=args.learning_rate, mavlink_endpoint=args.mavlink)',
        globals(), locals(),
        'profile_output.prof'
    )
