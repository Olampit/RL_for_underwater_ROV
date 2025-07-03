# environment.py

from pymavlink import mavutil
import numpy as np
import time
import subprocess
from joystick_input import FakeJoystick
import math
import random

from imu_reader import attitude_buffer, velocity_buffer, goal_buffer, raw_buffer

SERVO_MIN = 1100
SERVO_MAX = 1900
SERVO_IDLE = 1500

def input_to_pwm(value):
    if abs(value) < 0.05:
        return SERVO_IDLE
    pwm = SERVO_IDLE + (value * 400)
    return int(max(SERVO_MIN, min(SERVO_MAX, pwm)))

class ROVEnvironment:
    def __init__(self, action_map, connection, latest_imu):
        self.action_map = action_map
        self.connection = connection
        self.latest_imu = latest_imu
        self.joystick = FakeJoystick()
        self.target_velocity = self.joystick.get_target()

    def apply_action(self, action_idx):
        action = self.action_map[action_idx]
        for i in range(8):
            motor_label = f"motor{i+1}"
            thrust = action.get(motor_label, 0.0)
            pwm = input_to_pwm(thrust)
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                i + 1,
                pwm,
                0, 0, 0, 0, 0
            )

    from imu_reader import attitude_buffer, raw_buffer, goal_buffer
    
    


    def get_state(self, time_before_action):
        """
        Uses the time_before_action argument to select data from buffers.
        Returns an averaged, temporally-aligned state.
        """
        state = {}
        MAX_AGE = 0.02
        now = time.time()
        start_time = time_before_action

        raw_seq = raw_buffer.get_since(start_time, max_age=MAX_AGE)
        att_seq = attitude_buffer.get_since(start_time, max_age=MAX_AGE)
        goal_seq = goal_buffer.get_since(start_time, max_age=MAX_AGE)

        if not goal_seq:
            for k in ["goal_vx", "goal_vy", "goal_vz", "goal_roll", "goal_pitch", "goal_yaw"]:
                state[k] = 0.0
        else:
            goal_values = [g for _, g in goal_seq]
            state["goal_vx"] = np.mean([g.get("vx", 0.0) for g in goal_values])
            state["goal_vy"] = np.mean([g.get("vy", 0.0) for g in goal_values])
            state["goal_vz"] = np.mean([g.get("vz", 0.0) for g in goal_values])
            state["goal_roll"] = np.mean([g.get("roll", 0.0) for g in goal_values])
            state["goal_pitch"] = np.mean([g.get("pitch", 0.0) for g in goal_values])
            state["goal_yaw"] = np.mean([g.get("yaw", 0.0) for g in goal_values])

        if raw_seq:
            state["ax"] = np.mean([r.get("ax", 0.0) for _, r in raw_seq])
            state["ay"] = np.mean([r.get("ay", 0.0) for _, r in raw_seq])
            state["az"] = np.mean([r.get("az", 0.0) for _, r in raw_seq])
        else:
            state["ax"] = state["ay"] = state["az"] = 0.0

        if att_seq:
            rolls = [a.get("roll", 0.0) for _, a in att_seq]
            pitches = [a.get("pitch", 0.0) for _, a in att_seq]
            yaws = [a.get("yaw", 0.0) for _, a in att_seq]

            def wrap_angle(angle):
                return (angle + np.pi) % (2 * np.pi) - np.pi

            state["roll_error"] = wrap_angle(np.mean(rolls) - state["goal_roll"])
            state["pitch_error"] = wrap_angle(np.mean(pitches) - state["goal_pitch"])
            state["yaw_error"] = wrap_angle(np.mean(yaws) - state["goal_yaw"])

            state["rollspeed"] = np.mean([abs(a.get("rollspeed", 0.0)) for _, a in att_seq])
            state["pitchspeed"] = np.mean([abs(a.get("pitchspeed", 0.0)) for _, a in att_seq])
            state["yawspeed"] = np.mean([abs(a.get("yawspeed", 0.0)) for _, a in att_seq])
        else:
            state["roll_error"] = state["pitch_error"] = state["yaw_error"] = 0.0
            state["rollspeed"] = state["pitchspeed"] = state["yawspeed"] = 0.0

        return state



    



    def random_orientation_quat(self, max_angle_deg=15):
        max_angle_rad = math.radians(max_angle_deg)
        roll = random.uniform(-max_angle_rad, max_angle_rad)
        pitch = random.uniform(-max_angle_rad, max_angle_rad)
        yaw = random.uniform(-math.pi, math.pi)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        return {
            "x": sr * cp * cy - cr * sp * sy,
            "y": cr * sp * cy + sr * cp * sy,
            "z": cr * cp * sy - sr * sp * cy,
            "w": cr * cp * cy + sr * sp * sy,
        }

    def reset(self):
        # px = round(random.uniform(-1.0, 1.0), 2)
        # py = round(random.uniform(4999.0, 5001.0), 2)
        # pz = round(random.uniform(39.0, 41.0), 2)
        
        time_before_reset = time.time()
        
        px = 0
        py = 5000
        pz = 30
        
        # quat = self.random_orientation_quat(max_angle_deg=0)
        # qx, qy, qz, qw = quat["x"], quat["y"], quat["z"], quat["w"]
        
        # qx, qy, qz, qw = 0, 0, (np.sqrt(2))/2, (np.sqrt(2))/2
        
        odom_seq = velocity_buffer.get_last_n(1)
        if odom_seq:
            _, last_data = odom_seq[0]
            qx = last_data.get("qx", 0.0)
            qy = last_data.get("qy", 0.0)
            qz = last_data.get("qz", 0.0)
            qw = last_data.get("qw", 1.0)
        else:
            # Fallback in case buffer is empty
            qx, qy, qz, qw = 0.0, 0.0, np.sqrt(2)/2, np.sqrt(2)/2
            
        cmd = [
            "ros2", "service", "call",
            "/stonefish_ros2/respawn_robot",
            "stonefish_ros2/srv/Respawn",
            f"""{{name: 'bluerov',
            origin: {{
                position: {{x: {px}, y: {py}, z: {pz}}},
                orientation: {{x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}, w: {qw:.6f}}}
            }}}}"""
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self.get_state(time_before_reset)

    def stop_motors(self, connection):
        for servo in range(1, 9):
            connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                servo,
                1500,
                0, 0, 0, 0, 0
            )



    



    def compute_reward(self, start_of_action_time):
        CLIP = 100.0
        MAX_AGE = 0.02
        SCALE_VEL = 0.5
        SCALE_ANG = 1.0
        SCALE_SPIN = 1.0
        COEFF = 1.0
        SPIN_WEIGHT = 2.0

        def wrap(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        def penalty(x, scale):
            return -COEFF * np.log1p((x / scale) ** 2)

        now = time.time()
        start_time = start_of_action_time

        vel_seq = velocity_buffer.get_since(start_time, max_age=MAX_AGE)
        att_seq = attitude_buffer.get_since(start_time, max_age=MAX_AGE)
        goal_seq = goal_buffer.get_since(start_time, max_age=MAX_AGE)

        if not vel_seq or not att_seq or not goal_seq:
            return {"total": -CLIP, "reason": "missing data"}

        vel_values = [v for _, v in vel_seq]
        att_values = [a for _, a in att_seq]
        goal_values = [g for _, g in goal_seq]

        goal_vx = np.mean([g.get("vx", 0.0) for g in goal_values])
        goal_vy = np.mean([g.get("vy", 0.0) for g in goal_values])
        goal_vz = np.mean([g.get("vz", 0.0) for g in goal_values])
        goal_yaw = np.mean([g.get("yaw", 0.0) for g in goal_values])
        goal_pitch = np.mean([g.get("pitch", 0.0) for g in goal_values])
        goal_roll = np.mean([g.get("roll", 0.0) for g in goal_values])

        vx = np.mean([v.get("vx", 0.0) for v in vel_values])
        vy = np.mean([v.get("vy", 0.0) for v in vel_values])
        vz = np.mean([v.get("vz", 0.0) for v in vel_values])

        yaw = np.mean([a.get("yaw", 0.0) for a in att_values])
        pitch = np.mean([a.get("pitch", 0.0) for a in att_values])
        roll = np.mean([a.get("roll", 0.0) for a in att_values])

        # Mean angular speeds for spin penalties
        mean_yawspeed = np.mean([abs(a.get("yawspeed", 0.0)) for a in att_values])
        mean_pitchspeed = np.mean([abs(a.get("pitchspeed", 0.0)) for a in att_values])
        mean_rollspeed = np.mean([abs(a.get("rollspeed", 0.0)) for a in att_values])

        # Errors
        vx_error = vx - goal_vx
        vy_error = vy - goal_vy
        vz_error = vz - goal_vz
        yaw_error = wrap(yaw - goal_yaw)
        pitch_error = wrap(pitch - goal_pitch)
        roll_error = wrap(roll - goal_roll)

        # Scores: velocity
        vx_score = penalty(vx_error, SCALE_VEL)
        vy_score = penalty(vy_error, SCALE_VEL)
        vz_score = penalty(vz_error, SCALE_VEL)

        # Scores: attitude with spin penalties inside
        yaw_alignment = penalty(yaw_error, SCALE_ANG)
        yaw_spin = penalty(mean_yawspeed, SCALE_SPIN)
        yaw_score = yaw_alignment + SPIN_WEIGHT * yaw_spin

        pitch_alignment = penalty(pitch_error, SCALE_ANG)
        pitch_spin = penalty(mean_pitchspeed, SCALE_SPIN)
        pitch_score = pitch_alignment + SPIN_WEIGHT * pitch_spin

        roll_alignment = penalty(roll_error, SCALE_ANG)
        roll_spin = penalty(mean_rollspeed, SCALE_SPIN)
        roll_score = roll_alignment + SPIN_WEIGHT * roll_spin

        # Total reward
        total = (
            vx_score +
            vy_score +
            vz_score +
            yaw_score +
            pitch_score +
            roll_score
        )
        total = np.clip(total, -CLIP, CLIP)

        return {
            "total": total,
            "vx_score": vx_score,
            "vy_score": vy_score,
            "vz_score": vz_score,
            "yaw_score": yaw_score,
            "pitch_score": pitch_score,
            "roll_score": roll_score,
            "yaw_alignment": yaw_alignment,
            "yaw_spin": yaw_spin,
            "pitch_alignment": pitch_alignment,
            "pitch_spin": pitch_spin,
            "roll_alignment": roll_alignment,
            "roll_spin": roll_spin,
            "vx_error": vx_error,
            "vy_error": vy_error,
            "vz_error": vz_error,
            "yaw_error": yaw_error,
            "pitch_error": pitch_error,
            "roll_error": roll_error,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "goal_vx": goal_vx,
            "goal_vy": goal_vy,
            "goal_vz": goal_vz,
            "goal_yaw": goal_yaw,
            "goal_pitch": goal_pitch,
            "goal_roll": goal_roll,
        }












    def is_terminal(self):
        return False
    
    