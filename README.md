## Reinforcement Learning for Underwater ROVs

This repository contains the code developed during my internship at the Centre de Recherche sur les Communications (CRC) in Sophia-Antipolis. The project explores how Reinforcement Learning (RL) can be applied to **low-level motor control** of an underwater Remotely Operated Vehicle (ROV), using **real sensor input (IMU + odometry)** and **MAVLink-based actuation**.

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
├── sac/                        # Current SAC-based continuous control system
│ ├── run_training_sac.py       # Training loop (GUI/CLI-compatible)
│ ├── run_policy.py             # Runs trained SAC actor
│ ├── prefill_replay.py         # Replay buffer population
│ ├── rov_env_gym.py            # OpenAI Gym wrapper for ROV
│ ├── sac_agent.py              # SAC logic
│ ├── networks.py               # Actor/Critic networks
│ ├── replay_buffer.py          # Experience replay manager
│ ├── environment.py            # Real-time ROV interface
│ ├── imu_reader.py             # Threaded sensor listener (MAVLink + ROS2)
│ ├── joystick_input.py         # Simulated target velocities for RL
│ ├── sac_actor.pth             # Trained actor model
│ ├── sac_training_rewards.pdf  # Reward plot
│ └── replay_buffer.pkl         # Saved experience buffer
│
├── requirements.txt
├── README.md
└── run.sh
```

## Learning Methods

### Q-Learning (Tabular) (Deprecated !)

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

The ROV state is computed from **live asynchronous streams**, collected using:

- **MAVLink messages** (`ATTITUDE`): pitch, roll, yaw + angular rates.
- **ROS 2** (`/bluerov/navigator/odometry`): velocity (x, y, z).
- All messages are processed in real time using `imu_reader.py`, which maintains:
  - `attitude_buffer` — recent angular rates
  - `velocity_buffer` — recent linear velocities

These buffers are accessed by the environment to compute statistical features like:
- Mean / variance of angular rates
- Mean / variance of velocity
- Average velocity magnitude

## Dependencies

Install dependencies using:

```bash
python3 -m venv mavlink/venv
source mavlink/venv/bin/activate
pip install -r requirements.txt
```

You’ll also need:

    ROS 2 Jazzy installed and sourced

    A working BlueOS + MAVLink setup (real or simulated)



## How to Train and Evaluate

### Q-Learning:
```bash
cd Q_learning
./run.sh                            # Launches training
                                    # the run_policy.py file was bugged and as such has been removed for now
```

### SAC:
```bash
cd SAC
./run.sh                                # Starts SAC training through the GUI
! deprecated : python3 run_policy.py    # Runs the trained SAC actor
```

## Output Files

| File                       | Description                         |
| -------------------------- | ----------------------------------- |
| `sac_actor.pth`            | Saved actor model (PyTorch)         |
| `replay_buffer.pkl`        | Serialized experience replay buffer |
| `sac_training_rewards.pdf` | Reward curve per episode            |
| `profile_output.prof`      | CPU profiling data                  |


## Collaborators

This work was completed under the guidance of:

- Sébastien Travadel
- Luca Istrate
- Aymeric Cardot




## Hyperparameter Reference

This table lists all the key variables and hyperparameters that influence the reinforcement learning process, their role, and how changing them affects the training dynamics.

| **Variable**                  | **Location**               | **Description**                                                                 | **↑ Value =** More…                                                   | **↓ Value =** Less…                                                |
|-------------------------------|----------------------------|---------------------------------------------------------------------------------|-----------------------------------------------------------------------|--------------------------------------------------------------------|
| `gamma`                       | `SACAgent`                 | Discount factor for future rewards                                              | Long-term focus                                                       | Focuses on immediate reward                                        |
| `tau`                         | `SACAgent`                 | Soft update rate for target critic                                              | Stable, slower updates                                                | More reactive, less stable                                         |
| `alpha`                       | `SACAgent`                 | Entropy coefficient (exploration vs. exploitation)                              | More exploration                                                      | More deterministic policy                                          |
| `automatic_entropy_tuning`    | `SACAgent`                 | Automatically adjust alpha based on policy entropy                              | Dynamic exploration                                                   | Uses fixed alpha                                                   |
| `learning_rate`               | `train()`                  | Optimizer learning rate (actor & critic)                                        | Faster learning, but riskier                                          | Slower, more stable convergence                                    |
| `batch_size`                  | `train()`                  | Batch size for SAC updates                                                      | Better gradient estimate                                              | Noisier learning                                                   |
| `start_steps`                 | `train()`                  | Random exploration steps before learning starts                                 | Better buffer initialization                                          | Risk of learning from bad data                                     |
| `update_every`                | `train()`                  | Update frequency (in environment steps)                                         | More frequent learning                                                | Less frequent updates                                              |
| `reward_scale`                | `train()`, `prefill`       | Scale factor applied to rewards                                                 | Stronger gradients                                                    | May underutilize rewards                                           |
| `max_steps`                   | `train()`                  | Maximum steps per episode                                                       | Longer policy exploration                                             | Shorter episodes                                                   |
| `episodes`                    | `train()`                  | Total training episodes                                                         | More experience                                                       | Less experience, faster runs                                       |
| `checkpoint_every`            | `train()`                  | Save model every N steps                                                        | Frequent backups                                                      | Risk of losing progress                                            |
| `capacity`                    | `ReplayBuffer`             | Size of the replay buffer                                                       | More history, better sampling                                         | Smaller memory, more overwriting                                   |
| `hidden_dims`                 | `MLP`                      | Architecture of neural nets (e.g. `(128, 128)`)                                 | Higher capacity                                                       | Faster but less expressive                                         |
| `LOG_STD_MIN/MAX`             | `Actor`                    | Limits on log standard deviation                                                | Allows more/less uncertainty in sampling                              | Restricts stochasticity                                            |
| `vx_mean`, `std`, etc.        | `FakeJoystick`             | Target velocity goals and their tolerance                                       | Harder goals if std is small                                          | Looser objectives, less challenge                                  |
| `SPEED_UP`                    | `rov_env_gym`, `train()`   | Time acceleration factor                                                        | Faster training per wall time                                         | More realistic pacing                                              |
