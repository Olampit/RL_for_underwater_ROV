import asyncio
import aiohttp
from pymavlink import mavutil

SERVO_IDLE = 1500
SERVO_MIN = 1100
SERVO_MAX = 1900

def input_to_pwm(value):
    if abs(value) < 0.05:
        return SERVO_IDLE
    pwm = SERVO_IDLE + (value * 400)
    return int(max(SERVO_MIN, min(SERVO_MAX, pwm)))

def send_servo_command(conn, servo, pwm):
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0, servo, pwm, 0, 0, 0, 0, 0
    )
    print(f"[PWM] Servo {servo} → {pwm}")

async def executor_loop():
    conn = mavutil.mavlink_connection('udp:0.0.0.0:14550')
    conn.wait_heartbeat()
    print("[MAVLINK] Connected.")

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get("http://localhost:8080/state") as resp:
                    state = await resp.json()
                    thrust_pwm = input_to_pwm(state.get("thrust", 0.0))
                    yaw_pwm = input_to_pwm(state.get("yaw", 0.0))

                    send_servo_command(conn, 3, thrust_pwm)  # e.g. forward
                    send_servo_command(conn, 4, yaw_pwm)     # e.g. turn
            except Exception as e:
                print("[ERROR]", e)

            await asyncio.sleep(0.1)  # ~10Hz control loop

if __name__ == "__main__":
    asyncio.run(executor_loop())
