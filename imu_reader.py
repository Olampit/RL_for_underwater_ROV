#imu_reader.property
from pymavlink import mavutil
import threading
import traceback

# Messages we are interested in
imu_types = ['RAW_IMU', 'AHRS2', 'VIBRATION']

def start_imu_listener(connection, latest_imu):
    def imu_loop():
        try:
            print("[IMU] Starting listener thread...")

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
                    print("[IMU] No message received in last 5 seconds.")
                    continue

                msg_type = msg.get_type()
                print(f"[IMU] Received: {msg_type}")

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

    # Start thread
    threading.Thread(target=imu_loop, daemon=True).start()

