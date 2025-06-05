#imu_reader.py
from pymavlink import mavutil
import threading
import traceback

# ROS 2 imports
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np

# Messages we are interested in from MAVLink
imu_types = ['RAW_IMU', 'AHRS2', 'VIBRATION']

def start_imu_listener(connection, latest_imu):
    def imu_loop():
        try:
            print("[IMU] Starting MAVLink listener thread...")

            def update_imu_combined():
                raw = latest_imu.get("RAW_IMU", {})
                combined = {
                    "acc_x": raw.get("acc_x", 0.0),
                    "acc_y": raw.get("acc_y", 0.0),
                    "acc_z": raw.get("acc_z", 0.0),
                    "gyro_x": raw.get("gyro_x", 0.0),
                    "gyro_y": raw.get("gyro_y", 0.0),
                    "gyro_z": raw.get("gyro_z", 0.0),
                    "mag_x": raw.get("mag_x", 0.0),
                    "mag_y": raw.get("mag_y", 0.0),
                    "mag_z": raw.get("mag_z", 0.0),
                }
                latest_imu["IMU_COMBINED"] = combined

            while True:
                msg = connection.recv_match(type=imu_types, blocking=True, timeout=5)
                if msg is None:
                    print("[IMU] No MAVLink message in 5 seconds.")
                    continue

                msg_type = msg.get_type()
                try:
                    if msg_type == 'RAW_IMU':
                        latest_imu["RAW_IMU"] = {
                            "acc_x": getattr(msg, 'xacc', 0.0) / 1000.0,
                            "acc_y": getattr(msg, 'yacc', 0.0) / 1000.0,
                            "acc_z": getattr(msg, 'zacc', 0.0) / 1000.0,
                            "gyro_x": getattr(msg, 'xgyro', 0.0) / 1000.0,
                            "gyro_y": getattr(msg, 'ygyro', 0.0) / 1000.0,
                            "gyro_z": getattr(msg, 'zgyro', 0.0) / 1000.0,
                            "mag_x": getattr(msg, 'xmag', 0.0),
                            "mag_y": getattr(msg, 'ymag', 0.0),
                            "mag_z": getattr(msg, 'zmag', 0.0),
                        }
                        update_imu_combined()
                        latest_imu["imu_ready"] = True

                    elif msg_type == 'AHRS2':
                        latest_imu["AHRS2"] = {
                            "roll": getattr(msg, 'roll', 0.0),
                            "pitch": getattr(msg, 'pitch', 0.0),
                            "yaw": getattr(msg, 'yaw', 0.0),
                        }

                    elif msg_type == 'VIBRATION':
                        latest_imu["VIBRATION"] = {
                            "vibration_x": getattr(msg, 'vibration_x', 0.0),
                            "vibration_y": getattr(msg, 'vibration_y', 0.0),
                            "vibration_z": getattr(msg, 'vibration_z', 0.0),
                        }

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

                    # Store in latest_imu
                    latest_imu["ODOMETRY"] = {
                        "position": {
                            "x": msg.pose.pose.position.x,
                            "y": msg.pose.pose.position.y,
                            "z": msg.pose.pose.position.z,
                        },
                        "velocity": {
                            "x": velocity_x,
                            "y": velocity_y,
                            "z": velocity_z,
                            "magnitude": velocity_mag,
                            "average": average_velocity
                        }
                    }

                    # Print every Nth callback
                    self.odom_count += 1
                    if self.odom_count % self.print_every == 0:
                        print(f"[ODOM] Velocity: {velocity_mag:.2f} m/s | Avg: {average_velocity:.2f} m/s")

            node = OdomListener()
            #print("test2")
            rclpy.spin(node)
            node.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            print(f"[ROS2 ODOM ERROR] {e}")
            traceback.print_exc()

    threading.Thread(target=ros_spin, daemon=True).start()
