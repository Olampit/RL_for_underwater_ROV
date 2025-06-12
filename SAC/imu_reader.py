#imu_reader.py
from pymavlink import mavutil
import threading
import traceback
import time

# ROS 2 imports
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np

from imu_buffer import IMUBuffer


attitude_buffer = IMUBuffer(max_seconds=1.0, frequency=400)
velocity_buffer = IMUBuffer(max_seconds=1.0, frequency=400)


# Messages we are interested in from MAVLink
imu_types = ['ATTITUDE', 'VIBRATION']

def start_imu_listener(connection, latest_imu):
    def imu_loop():
        try:
            print("[IMU] Starting MAVLink listener thread...")

            while True:
                msg = connection.recv_match(type=imu_types, blocking=True, timeout=5)
                if msg is None:
                    print("[IMU] No MAVLink message in 5 seconds.")
                    continue

                msg_type = msg.get_type()
                try:
                    if msg_type == 'ATTITUDE': 
                        imu_data = {
                            "pitch": getattr(msg, 'pitch', 0.0),
                            "pitchspeed": getattr(msg, 'pitchspeed', 0.0),
                            "roll": getattr(msg, 'roll', 0.0),
                            "rollspeed": getattr(msg, 'rollspeed', 0.0),
                            "yaw": getattr(msg, 'yaw', 0.0),
                            "yawspeed": getattr(msg, 'yawspeed', 0.0),
                        }
                        attitude_buffer.add(time.time(), imu_data)

                except AttributeError as e:
                    print(f"[IMU] Missing attribute: {e}")
                except Exception as e:
                    print(f"[IMU] Unexpected error: {e}")
                    traceback.print_exc()

        except Exception as e:
            print(f"[IMU THREAD ERROR] {e}")
            traceback.print_exc()

    # Start MAVLink listener thread
    threading.Thread(target=imu_loop, daemon=True).start()

    # Start ROS2 Odometry listener
    def ros_spin():
        try:
            rclpy.init()
            class OdomListener(Node):
                def __init__(self):
                    super().__init__('odom_listener')
                    self.subscription = self.create_subscription(
                        Odometry,
                        '/bluerov/navigator/odometry',
                        self.odom_callback,
                        10
                    )
                    self.velocity_history = []
                    self.print_every = 1000
                    self.odom_count = 0
                    #print("odometry started")

                def odom_callback(self, msg):
                    velocity_x = msg.twist.twist.linear.x
                    velocity_y = msg.twist.twist.linear.y
                    velocity_z = msg.twist.twist.linear.z

                    velocity_mag = np.linalg.norm([velocity_x, velocity_y, velocity_z])

                    # Save velocity history for average calculation
                    self.velocity_history.append(velocity_mag)
                    if len(self.velocity_history) > 100:
                        self.velocity_history.pop(0)

                    average_velocity = np.mean(self.velocity_history)

                    velocity_data = {
                        "vx": velocity_x,
                        "vy": velocity_y,
                        "vz": velocity_z,
                        "mag": velocity_mag,
                        "avg": average_velocity
                    }
                    velocity_buffer.add(time.time(), velocity_data)


                    
            node = OdomListener()
            #print("test2")
            rclpy.spin(node)
            node.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            print(f"[ROS2 ODOM ERROR] {e}")
            traceback.print_exc()

    threading.Thread(target=ros_spin, daemon=True).start()
