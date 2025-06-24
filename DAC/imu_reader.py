# imu_reader.py

from pymavlink import mavutil
import threading
import traceback
import time

# ROS 2 imports
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
import numpy as np

from imu_buffer import IMUBuffer

attitude_buffer = IMUBuffer(max_seconds=1.0, frequency=400)
velocity_buffer = IMUBuffer(max_seconds=1.0, frequency=400)

# Shutdown control
stop_event = threading.Event()
ros_thread = None
imu_thread = None

imu_types = ['ATTITUDE', 'VIBRATION']

def start_imu_listener(connection, latest_imu):
    """
    Starts MAVLink attitude listener and ROS 2 odometry subscriber in separate threads.
    """
    def imu_loop():
        try:
            print("[IMU] Starting MAVLink listener thread...")
            print(time.time())

            while not stop_event.is_set():
                msg = connection.recv_match(type=imu_types, blocking=True, timeout=1)
                if msg is None:
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

                def odom_callback(self, msg):
                    velocity_x = msg.twist.twist.linear.x
                    velocity_y = msg.twist.twist.linear.y
                    velocity_z = msg.twist.twist.linear.z
                    velocity_mag = np.linalg.norm([velocity_x, velocity_y, velocity_z])

                    
                    
                    self.velocity_history.append(velocity_mag)
                    if len(self.velocity_history) > 100:
                        self.velocity_history.pop(0)

                    average_velocity = np.mean(self.velocity_history)
                    velocity_data = {
                        "vx": velocity_x,
                        "vy": velocity_y,
                        "vz": velocity_z,
                        "mag": velocity_mag,
                        "avg": average_velocity,
                        "qx": msg.pose.pose.orientation.x,
                        "qy": msg.pose.pose.orientation.y,
                        "qz": msg.pose.pose.orientation.z,
                        "qw": msg.pose.pose.orientation.w
                    }
                    velocity_buffer.add(time.time(), velocity_data)

            node = OdomListener()
            executor = SingleThreadedExecutor()
            executor.add_node(node)

            try:
                while rclpy.ok() and not stop_event.is_set():
                    executor.spin_once(timeout_sec=0.1)
            finally:
                executor.shutdown()
                node.destroy_node()
                rclpy.shutdown()

        except Exception as e:
            print(f"[ROS2 ODOM ERROR] {e}")
            traceback.print_exc()

    global imu_thread, ros_thread
    imu_thread = threading.Thread(target=imu_loop, daemon=True)
    ros_thread = threading.Thread(target=ros_spin, daemon=True)

    imu_thread.start()
    ros_thread.start()
