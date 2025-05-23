#Executor that sends commands to the motors
from pymavlink import mavutil
import time

SERVO_MIN = 1100
SERVO_MAX = 1900
SERVO_IDLE = 1500

def input_to_pwm(value):
    if abs(value) < 0.05:
        return SERVO_IDLE
    pwm = SERVO_IDLE + (value * 400)
    return int(max(SERVO_MIN, min(SERVO_MAX, pwm)))

def send_servo_command(connection, servo_number, pwm_value):
    print(connection.target_system, connection.target_component)
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        servo_number,
        pwm_value,
        0, 0, 0, 0, 0
    )
    print(f"Sent PWM {pwm_value} to servo {servo_number}")


print("Connecting...")
connection = mavutil.mavlink_connection('udp:0.0.0.0:14550')
connection.wait_heartbeat()
print(f"Connected to system {connection.target_system}, component {connection.target_component}")

# Example: set servo 3 to 75% forward
thrust_input = 0.04  # range from -1.0 to 1.0
pwm = input_to_pwm(thrust_input)
send_servo_command(connection, servo_number=3, pwm_value=pwm)


