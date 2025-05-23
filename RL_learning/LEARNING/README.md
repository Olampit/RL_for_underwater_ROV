# README
## RL for underwater robots

this is the readme for the project I developped during my stage at the CRC, in Sophia-Antipolis. 


Project structure : 

RL/
├── run.sh                  # Lance l’entraînement avec venv
├── imu_reader.py           # Lit les données IMU (pour logging ou état enrichi)
├── executor.py             # Lit l’état HTTP et envoie les PWM via MAVLink
├── state_server.py         # aiohttp : reçoit les états demandés (depuis RL agent)
├── q_agent.py              # Agent Q-learning
├── environment.py          # Interface async entre agent et ROV (HTTP)
├── run_training.py         # Entraîne l’agent (appelé depuis run.sh)
├── run_policy.py           # Runs the policy on the robot. 
└── mavlink                 # setup for pymavlink and the venv
    └──venv







dependencies : 
numpy
time
aiohttp
asyncio
q_agent
environment
random
pickle
pymavlink


## Collaborators
Sébastien TravLuca Istrate, Aymeric Cardot, 