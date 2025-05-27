# state_server.py

import asyncio
import aiohttp
import signal
import time
from aiohttp import web
from pymavlink import mavutil
from imu_reader import start_imu_reader

SERVO_MIN = 1100
SERVO_MAX = 1900
SERVO_IDLE = 1500

def input_to_pwm(value):
    if abs(value) < 0.05:
        return SERVO_IDLE
    pwm = SERVO_IDLE + (value * 400)
    return int(max(SERVO_MIN, min(SERVO_MAX, pwm)))


class ROVController:
    def __init__(self):
        self.state = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "thrust": 0.0, "yaw": 0.0,
            "depth": 0.0, "pressure_abs": 1013.25,
            "desired_speed": 0.0,
            "acc_x": 0.0, "acc_y": 0.0, "acc_z": 0.0,
            "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
            "mag_x": 0.0, "mag_y": 0.0, "mag_z": 0.0,
            "imu_ready": False
        }

        self.imu_data = {}
        print("Connecting to ROV via MAVLink...")
        self.connection = mavutil.mavlink_connection('udp:0.0.0.0:14550')
        print("Waiting for ROV heartbeat...")
        self.connection.wait_heartbeat()
        print(f"Connected to system {self.connection.target_system}, component {self.connection.target_component}")
        print("Starting IMU reader...")
        start_imu_reader(self.imu_data, self.connection)

    async def run_background_tasks(self):
        print("Waiting for first IMU data...")
        start_time = time.time()
        while not self.imu_data.get("imu_ready", False):
            if time.time() - start_time > 10:
                print("IMU not ready after 10 seconds, proceeding anyway...")
                break
            await asyncio.sleep(0.1)

        print("IMU ready. Starting state sync loop.")

        while True:
            imu = self.imu_data.get("IMU_COMBINED", {})
            self.state.update({
                "acc_x": imu.get("acc_x", 0.0),
                "acc_y": imu.get("acc_y", 0.0),
                "acc_z": imu.get("acc_z", 0.0),
                "gyro_x": imu.get("gyro_x", 0.0),
                "gyro_y": imu.get("gyro_y", 0.0),
                "gyro_z": imu.get("gyro_z", 0.0),
                "mag_x": imu.get("mag_x", 0.0),
                "mag_y": imu.get("mag_y", 0.0),
                "mag_z": imu.get("mag_z", 0.0),
                "imu_ready": self.imu_data.get("imu_ready", False)
            })
            await asyncio.sleep(0.1)

    async def update_state(self, request):
        data = await request.json()
        print(f"[RECEIVED COMMAND] /update {data}")
        self.state.update(data)
        return web.json_response({"status": "ok", "received": data})

    async def joystick_input(self, request):
        data = await request.json()
        print(f"[RECEIVED JOYSTICK] /joystick {data}")
        if "desired_speed" in data:
            self.state["desired_speed"] = data["desired_speed"]
        return web.json_response({"status": "ok", "received": data})

    async def get_state(self, request):
        return web.json_response(self.state)

    async def apply_action(self, request):
        data = await request.json()
        print(f"[RECEIVED ACTION] /apply_action {data}")

        for motor_key, value in data.items():
            motor_num = int(motor_key.replace("motor", ""))
            pwm = input_to_pwm(value)
            print(f"Sending PWM {pwm} to motor {motor_num}")
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,  # confirmation
                motor_num,
                pwm,
                0, 0, 0, 0, 0
            )

        return web.json_response({"status": "applied", "command": data})

    def routes(self):
        return [
            web.post('/update', self.update_state),
            web.post('/joystick', self.joystick_input),
            web.post('/apply_action', self.apply_action),
            web.get('/state', self.get_state)
        ]


async def main():
    controller = ROVController()
    app = web.Application()
    app.add_routes(controller.routes())

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()

    print("ROV Controller API running on http://localhost:8080")

    #Lancer les tâches asynchrones après que le serveur écoute
    asyncio.create_task(controller.run_background_tasks())

    # Gestion des signaux pour arrêt propre
    stop_event = asyncio.Event()

    def handle_sigint():
        print("Received SIGINT, shutting down...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, handle_sigint)
    loop.add_signal_handler(signal.SIGTERM, handle_sigint)

    await stop_event.wait()
    await runner.cleanup()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[ERROR] Server failed to start: {e}")
