#the environment for the learning agent. This links the aiohttp server and the q-learning agent, currently. 

import aiohttp
import numpy as np
from pymavlink import mavutil

SERVO_MIN = 1100
SERVO_MAX = 1900
SERVO_IDLE = 1500

def input_to_pwm(value):
    if abs(value) < 0.05:
        return SERVO_IDLE
    pwm = SERVO_IDLE + (value * 400)
    return int(max(SERVO_MIN, min(SERVO_MAX, pwm)))


class ROVEnvironment:
    def __init__(self, action_map, api_url="http://localhost:8080"):
        self.action_map = action_map
        self.api_url = api_url
        
        self.connection = mavutil.mavlink_connection('udp:0.0.0.0:14550')
        print("Waiting for ROV heartbeat...")
        self.connection.wait_heartbeat()
        print(f"Connected to system {self.connection.target_system}, component {self.connection.target_component}")



    async def get_state(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_url}/state") as resp:
                return await resp.json()  # doit être un dict avec .get("depth")


    def state_to_index(self, state):
        # à adapter selon tes états
        return int(state["depth"] // 0.5)  # ex: discretisation naïve

    """ async def apply_action(self, action_index):
        command = self.action_map[action_index]
        print(f"[ACTION] Sending: {command}")
        async with aiohttp.ClientSession() as session:
            await session.post("http://localhost:8080/update", json=command) """
            
    async def apply_action(self, action_index):
        command = self.action_map[action_index]
        print(f"[ACTION] Sending: {command}")
        
        # Send to your HTTP server (simulated environment)
        async with aiohttp.ClientSession() as session:
            await session.post(f"{self.api_url}/update", json=command)

        # Send to real motors (through MAVLink)
        for motor_key, value in command.items():
            motor_num = int(motor_key.replace("motor", ""))
            pwm = input_to_pwm(value)
            print(f"Sending PWM {pwm} to motor {motor_num}")
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                motor_num,
                pwm,
                0, 0, 0, 0, 0
            )


    def compute_reward(self, state):
        # par exemple : rester à 1m de profondeur
        target = 1.0
        error = abs(state["depth"] - target)
        return -error
