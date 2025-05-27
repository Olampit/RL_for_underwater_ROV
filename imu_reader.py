#imu_reader.py

from pymavlink import mavutil
import threading
import time

def start_imu_reader(latest_imu):
    def imu_loop():
        connection = mavutil.mavlink_connection('udp:0.0.0.0:14550')

        try:
            print("Waiting for MAVLink heartbeat...")
            connection.wait_heartbeat(timeout=10)
            print(f"Connected to system {connection.target_system}, component {connection.target_component}")
        except Exception as e:
            print(f"Heartbeat timeout: {e}")
            return

        imu_types = ['RAW_IMU', 'AHRS', 'AHRS2', 'VIBRATION']

 
        #Wait for first message
        msg = connection.recv_match(type=imu_types, blocking=True, timeout=5)
        if msg is None:
            print("No IMU message received in last 5 seconds.")
        #print(f"Got: {msg.get_type()}")


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
            #print(f"[IMU UPDATE] acc=({combined['acc_x']}, {combined['acc_y']}, {combined['acc_z']})")

        while True:
            msg = connection.recv_match(type=imu_types, blocking=True, timeout=5)
            if msg is None:
                print("No IMU message received in last 5 seconds.")
                continue


            msg_type = msg.get_type()
            #print(f"Received: {msg_type}")
            #print(msg)

            try:
                if msg_type == 'RAW_IMU':
                    latest_imu["RAW_IMU"] = {
                        "acc_x": getattr(msg, 'xacc', 0.0),
                        "acc_y": getattr(msg, 'yacc', 0.0),
                        "acc_z": getattr(msg, 'zacc', 0.0),
                        "gyro_x": getattr(msg, 'xgyro', 0.0),
                        "gyro_y": getattr(msg, 'ygyro', 0.0),
                        "gyro_z": getattr(msg, 'zgyro', 0.0),
                        "mag_x": getattr(msg, 'xmag', 0.0),
                        "mag_y": getattr(msg, 'ymag', 0.0),
                        "mag_z": getattr(msg, 'zmag', 0.0),
                    }
                    latest_imu["imu_ready"] = True
                    update_imu_combined()

                elif msg_type == 'AHRS':
                    latest_imu["AHRS"] = {
                        "omegaIx": msg.omegaIx,
                        "omegaIy": msg.omegaIy,
                        "omegaIz": msg.omegaIz,
                        "accel_weight": {msg.accel_weight},
                        "renorm_val": {msg.renorm_val},
                        "error_rp": {msg.error_rp},
                        "error_yaw": {msg.error_yaw}
                    }

                elif msg_type == 'AHRS2':
                    latest_imu["AHRS2"] = {
                        "roll": msg.roll,
                        "pitch": msg.pitch,
                        "yaw": msg.yaw
                    }

                elif msg_type == 'VIBRATION':
                    latest_imu["VIBRATION"] = {
                        "vibration_x": msg.vibration_x,
                        "vibration_y": msg.vibration_y,
                        "vibration_z": msg.vibration_z
                    }

            except AttributeError as e:
                print(f"Skipped message due to missing attribute: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")

            time.sleep(0.01)

    threading.Thread(target=imu_loop, daemon=True).start()
