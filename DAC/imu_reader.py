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

# Buffers
attitude_buffer = IMUBuffer(max_seconds=1.0, frequency=400)
velocity_buffer = IMUBuffer(max_seconds=1.0, frequency=400)
goal_buffer = IMUBuffer(max_seconds=1.0, frequency=400)

# Shared state for synchronization
latest_att_data = {}
latest_vel_data = {}
joystick_ref = None  # Set externally when starting

# Shutdown control
stop_event = threading.Event()
ros_thread = None
imu_thread = None

imu_types = ['ATTITUDE', 'VIBRATION']


def log_synchronized_frame(att_data, vel_data, joystick):
    t = time.time()

    # Log state
    attitude_buffer.add(t, att_data)
    velocity_buffer.add(t, vel_data)

    # Log goal
    goal = joystick.get_target()
    goal_data = {
        "vx": goal["vx"]["mean"],
        "vy": goal["vy"]["mean"],
        "vz": goal["vz"]["mean"],
        "yaw_rate": goal["yaw_rate"]["mean"],
        "pitch_rate": goal["pitch_rate"]["mean"],
        "roll_rate": goal["roll_rate"]["mean"],
    }
    goal_buffer.add(t, goal_data)


def synchronized_logging_loop():
    print("[SYNC] Starting synchronized logger...")
    while not stop_event.is_set():
        if latest_att_data and latest_vel_data and joystick_ref:
            log_synchronized_frame(
                latest_att_data.copy(),
                latest_vel_data.copy(),
                joystick_ref
            )
        time.sleep(1 / 400.0)  # match logging frequency


def start_imu_listener(connection, latest_imu, joystick):
    """
    Starts MAVLink and ROS 2 listeners and launches synchronized logger.
    """
    global imu_thread, ros_thread, joystick_ref
    joystick_ref = joystick

    def imu_loop():
        try:
            print("[IMU] Starting MAVLink listener thread...")
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
                        latest_att_data.update(imu_data)
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
                    latest_vel_data.update(velocity_data)

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

    # Start threads
    imu_thread = threading.Thread(target=imu_loop, daemon=True)
    ros_thread = threading.Thread(target=ros_spin, daemon=True)
    sync_thread = threading.Thread(target=synchronized_logging_loop, daemon=True)

    imu_thread.start()
    ros_thread.start()
    sync_thread.start()
