# README
## RL for underwater robots

this is the readme for the project I developped during my stage at the CRC, in Sophia-Antipolis. 


Project structure : 

RL/
├── run.sh                  # Lance l’entraînement avec venv<br />
├── imu_reader.py           # Lit les données IMU (pour logging ou état enrichi)<br />
├── executor.py             # Lit l’état HTTP et envoie les PWM via MAVLink<br />
├── state_server.py         # aiohttp : reçoit les états demandés (depuis RL agent)<br />
├── q_agent.py              # Agent Q-learning<br />
├── environment.py          # Interface async entre agent et ROV (HTTP)<br />
├── run_training.py         # Entraîne l’agent (appelé depuis run.sh)<br />
├── run_policy.py           # Runs the policy on the robot. <br />
└── mavlink                 # setup for pymavlink and the venv<br />
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