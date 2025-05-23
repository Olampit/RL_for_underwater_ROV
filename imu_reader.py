# imu_reader.py
from pymavlink import mavutil

# port used to connect to the rov
#connection = mavutil.mavlink_connection('udp:192.168.2.2:14550') 
#this one did not work, since it is on myself ?


#listening to everything ? should ? work
connection = mavutil.mavlink_connection('udp:0.0.0.0:14550')


 
#waiting for a heartbeat is mandatory
print("Attente du heartbeat...")
connection.wait_heartbeat()
print(f"Connecté à système {connection.target_system}, composant {connection.target_component}")

# read IMU
print("Lecture des données IMU...")
while True:
    msg = connection.recv_match(type='RAW_IMU', blocking=True)
    if msg:
        print(f"Acc: x={msg.xacc} y={msg.yacc} z={msg.zacc} | "
              f"Gyro: x={msg.xgyro} y={msg.ygyro} z={msg.zgyro} | "
              f"Mag: x={msg.xmag} y={msg.ymag} z={msg.zmag}")
