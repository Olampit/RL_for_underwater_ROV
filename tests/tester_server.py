import time
import requests

URL = "http://localhost:8080/update"

def send_command(thrust, yaw):
    data = {
        "thrust": thrust,
        "yaw": yaw
    }
    response = requests.post(URL, json=data)
    print(f"[TEST] Sent: {data}, Response: {response.json()}")

while True:
    # Go forward for 2 seconds
    send_command(thrust=0.8, yaw=0.0)
    time.sleep(2)

    # Stop for 3 seconds
    send_command(thrust=0.0, yaw=0.0)
    time.sleep(3)
