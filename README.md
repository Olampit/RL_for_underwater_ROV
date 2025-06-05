# Reinforcement Learning for Underwater ROVs

This repository contains the code developed during my internship at the Centre de Recherche sur les Communications (CRC) in Sophia-Antipolis. The project explores how Reinforcement Learning (RL) can be applied to low-level motor control for an underwater Remotely Operated Vehicle (ROV), using real sensor input and MAVLink-based actuation.

## Project Structure

```
RL/
├── q_learning/
│   ├── run_training.py         # Q-learning training loop
│   ├── run_policy.py           # Runs a learned Q-learning policy
│   ├── q_agent.py              # Discrete Q-learning agent
│   ├── environment.py          # Environment interface (Q-learning version)
│   ├── imu_reader.py           # Sensor listener (shared)
│   ├── state_server.py         # API-based state interface (Q-learning specific)
│   ├── executor.py             # Command sender via MAVLink (Q-learning specific)
│   └── q_table.pkl             # Saved Q-table
│
├── sac/
│   ├── run_training_sac.py     # SAC training loop
│   ├── run_policy.py           # Runs a learned SAC policy
│   ├── rov_env_gym.py          # Gym wrapper for the ROV environment
│   ├── sac_agent.py            # SAC coordination logic
│   ├── networks.py             # Actor and Critic networks
│   ├── replay_buffer.py        # Experience replay buffer
│   ├── environment.py          # Environment interface (SAC version)
│   ├── imu_reader.py           # Sensor listener (shared)
│   ├── sac_actor.pth           # Saved SAC policy
│   └── sac_training_rewards.pdf # Training curve
│
├── README.md                   # Project overview
└── run.sh                      # Setup and Q-learning launcher
```

## Learning Methods

### Q-Learning (Tabular)

- Uses discrete states created by binning sensor values (e.g., pitch, roll, velocity).
- Actions consist of 8 motor thrust combinations, discretized into a small set (e.g., [-1, 0, 1]).
- A Q-table maps (state, action) pairs to expected rewards.
- Epsilon-greedy exploration balances learning and exploitation.

Goal: go forward at certain depth

Limitations:
- Poor scalability to complex or continuous environments
- High variance depending on initial action samples

### Soft Actor-Critic (SAC)

- Uses continuous action control: each of the 8 motors receives a float in [-1.0, 1.0].
- SAC is a modern off-policy algorithm using stochastic policies and entropy maximization.
- All components are implemented in PyTorch.
- The environment is wrapped to comply with the OpenAI Gym API.

Advantages:
- Better generalization across continuous action spaces
- Smoother and more adaptive behaviors
- Sample-efficient updates using replay buffer

## Sensor Integration

The ROV's state is estimated using the following inputs:
- IMU: Acceleration, gyroscope, magnetometer (via MAVLink)
- AHRS2: Orientation (pitch, roll, yaw)
- Odometry: Position and velocity (via ROS2 /nav_msgs/Odometry)

These are combined in a shared `latest_imu` dictionary used by both Q-learning and SAC agents.

## Dependencies

Install dependencies using:

```bash
python3 -m venv mavlink/venv
source mavlink/venv/bin/activate
pip install -r requirements.txt
```

Requirements:
- numpy
- torch
- gym
- pymavlink
- matplotlib
- rclpy

## How to Train and Evaluate

### Q-Learning:
```bash
cd Q_learning
./run.sh          # Launches training
# the run_policy.py file was bugged and as such has been removed for now
```

### SAC:
```bash
cd SAC
./run.sh    # Starts SAC training
python3 run_policy.py          # Runs the trained SAC actor
```

## Output Files

- `q_table.pkl`: Q-table (Q-learning)
- `training_summary.pdf`: Learning curve
- `sac_actor.pth`: Actor model (SAC)
- `sac_training_rewards.pdf`: Learning curve

## Collaborators

This work was completed under the guidance of:

- Sébastien Travadel
- Luca Istrate
- Aymeric Cardot
