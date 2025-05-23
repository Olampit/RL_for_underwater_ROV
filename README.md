# README
## Reinforcement Learning for Underwater ROVs

This repository containss the code developed during my internship at the Centre de Recherche sur les Communications (CRC) in Sophia-Antipolis, focused on applying Reinforcement Learning (RL) to control an underwater Remotely Operated Vehicle (ROV) using real sensor data and simulated feedback.


## Project structure : 

RL/<br />
├── run.sh               # Script to set up the virtual environment and launch training<br />
├── imu_reader.py        # Reads IMU data (accelerometer, gyroscope, magnetometer)<br />
├── executor.py          # Reads ROV state via HTTP and sends motor PWM commands via MAVLink<br />
├── state_server.py      # aiohttp server: receives and returns state information for the agent<br />
├── q_agent.py           # Q-learning agent implementation<br />
├── environment.py       # Async environment wrapper interfacing agent and ROV<br />
├── run_training.py      # Main training loop for the RL agent<br />
├── run_policy.py        # Applies the learned policy on the physical ROV<br />
└── mavlink/             # Directory for MAVLink setup and virtual environment<br />
    └── venv/            # Python virtual environment<br />

## Description

This project explores online reinforcement learning for low-level motor control of an underwater robot. The agent interacts with its environment via an API that mirrors ROV dynamics and sends real commands to the robot using the MAVLink protocol.

Key goals:

    Apply Q-learning to a real robotic system.

    Experiment with reward shaping and state discretization.

    Integrate real-time pressure and IMU sensor data into the learning loop. Maybe DVL-based data as well.

    Create a reusable and modular Python interface for ROV training and deployment.

    Apply more advanced algorithms to get better results. 






## Dependencies : 

To run this project, make sure to install the following Python packages (you can use requirements.txt or a virtual environment):

    numpy

    aiohttp

    asyncio

    pymavlink

    pickle

Some modules like random and time are part of the Python standard library.

You can set up your environment with:

python3 -m venv mavlink/venv
source mavlink/venv/bin/activate
pip install -r requirements.txt  # if available


## How to Run

To train the agent:
./run.sh

To apply a trained policy on the real robot:
python run_policy.py

To monitor the IMU sensor:
python imu_reader.py


## Current Learning Method

This project uses a simple Q-learning algorithm with epsilon-greedy exploration. The environment is discrete and defined by state bins derived from depth, pressure, and other sensor inputs. The agent’s goal is to reach and maintain a stable target depth.

Upcoming extensions could include:

    Support for continuous state-action spaces via Deep Q-Networks (DQN)

    Improved reward functions based on sensor fusion

    Sim-to-real transfer learning using synthetic data

## Collaborators

This project was made possible with guidance and support from:

    Sébastien Travadel

    Luca Istrate

    Aymeric Cardot