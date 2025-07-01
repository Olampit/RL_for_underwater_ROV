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

    def get_state(self):
        """
        Builds an averaged, temporally-aligned state based on buffered sensor data.
        Uses new data since the last get_state() call, aligning IMU/goal in time.
        """
        state = {}
        now = time.time()
        max_age = 0.10

        if not hasattr(self, "last_obs_time"):
            self.last_obs_time = now - max_age

        raw_seq = raw_buffer.get_since(self.last_obs_time, max_age=max_age)
        att_seq = attitude_buffer.get_since(self.last_obs_time, max_age=max_age)
        goal_seq = goal_buffer.get_since(self.last_obs_time, max_age=max_age)

        self.last_obs_time = now

        if not goal_seq:
            for k in ["goal_vx", "goal_vy", "goal_vz", "goal_roll", "goal_pitch", "goal_yaw"]:
                state[k] = 0.0
        else:
            goal_times, goal_values = zip(*goal_seq)
            mean_goal = {}
            for key in ["vx", "vy", "vz", "roll", "pitch", "yaw"]:
                mean_goal[key] = np.mean([g.get(key, 0.0) for g in goal_values])
            state["goal_vx"] = mean_goal["vx"]
            state["goal_vy"] = mean_goal["vy"]
            state["goal_vz"] = mean_goal["vz"]
            state["goal_roll"] = mean_goal["roll"]
            state["goal_pitch"] = mean_goal["pitch"]
            state["goal_yaw"] = mean_goal["yaw"]

        if raw_seq:
            axs = [r.get("ax", 0.0) for _, r in raw_seq]
            ays = [r.get("ay", 0.0) for _, r in raw_seq]
            azs = [r.get("az", 0.0) for _, r in raw_seq]
            state["ax"] = np.mean(axs)
            state["ay"] = np.mean(ays)
            state["az"] = np.mean(azs)
        else:
            state["ax"] = state["ay"] = state["az"] = 0.0

        if att_seq:
            rolls = [a.get("roll", 0.0) for _, a in att_seq]
            pitches = [a.get("pitch", 0.0) for _, a in att_seq]
            yaws = [a.get("yaw", 0.0) for _, a in att_seq]
            state["roll"] = np.mean(rolls)
            state["pitch"] = np.mean(pitches)
            state["yaw"] = np.mean(yaws)
        else:
            state["roll"] = state["pitch"] = state["yaw"] = 0.0

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
        return self.get_state()

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



    



    def compute_reward(self, use_mean=True):
        TRACKING_WEIGHT = 1.0
        STABILITY_WEIGHT = 0.01
        ANGULAR_DEVIATION_WEIGHT = 2.0
        CLIP = 100.0
        MAX_AGE = 0.5  # increase time window

        def wrap_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        now = time.time()
        if not hasattr(self, "last_reward_time"):
            self.last_reward_time = now - MAX_AGE

        vel_seq = velocity_buffer.get_since(self.last_reward_time, max_age=MAX_AGE)
        att_seq = attitude_buffer.get_since(self.last_reward_time, max_age=MAX_AGE)
        goal_seq = goal_buffer.get_since(self.last_reward_time - 0.2, max_age=MAX_AGE + 0.2)

        self.last_reward_time = now

        if not vel_seq or not att_seq or len(goal_seq) < 2:
            return {"total": -CLIP, "reason": "missing data"}

        # --- Interpolate goal trajectory ---
        goal_times, goal_values = zip(*goal_seq)
        goal_fields = ["vx", "vy", "vz", "roll", "pitch", "yaw"]
        goal_interp = {k: np.interp(
            [t for t, _ in vel_seq], goal_times, [g.get(k, 0.0) for g in goal_values]
        ) for k in goal_fields}

        vx_errors, vy_errors, vz_errors = [], [], []
        yaw_errors, pitch_errors, roll_errors = [], [], []

        for i, ((_, vel), (_, att)) in enumerate(zip(vel_seq, att_seq)):
            vx_errors.append(vel.get("vx", 0.0) - goal_interp["vx"][i])
            vy_errors.append(vel.get("vy", 0.0) - goal_interp["vy"][i])
            vz_errors.append(vel.get("vz", 0.0) - goal_interp["vz"][i])
            yaw_errors.append(att.get("yaw", 0.0) - goal_interp["yaw"][i])
            pitch_errors.append(att.get("pitch", 0.0) - goal_interp["pitch"][i])
            roll_errors.append(att.get("roll", 0.0) - goal_interp["roll"][i])

        def shaped_penalty(err, scale, coeff):
            norm_err = err / scale
            return -coeff * np.log1p(norm_err ** 2)

        def compute_score(errs, scale, coeff):
            if not errs:
                return 0.0
            value = np.mean(errs) if use_mean else errs[-1]
            return shaped_penalty(value, scale, coeff)

        V_SCALE = 0.5
        R_SCALE = 1.0
        COEFF_V = 1.0
        COEFF_A = 1.0

        vx_score = compute_score(vx_errors, V_SCALE, COEFF_V)
        vy_score = compute_score(vy_errors, V_SCALE, COEFF_V)
        vz_score = compute_score(vz_errors, V_SCALE, COEFF_V)
        yaw_score = compute_score(yaw_errors, R_SCALE, COEFF_A)
        pitch_score = compute_score(pitch_errors, R_SCALE, COEFF_A)
        roll_score = compute_score(roll_errors, R_SCALE, COEFF_A)

        tracking_total = (
            vx_score + vy_score + vz_score +
            yaw_score + pitch_score + roll_score
        ) * TRACKING_WEIGHT

        yawspeeds = np.array([a.get("yawspeed", 0.0) for _, a in att_seq])
        pitchspeeds = np.array([a.get("pitchspeed", 0.0) for _, a in att_seq])
        rollspeeds = np.array([a.get("rollspeed", 0.0) for _, a in att_seq])

        yaw_dev = np.mean(np.abs(yawspeeds))
        pitch_dev = np.mean(np.abs(pitchspeeds))
        roll_dev = np.mean(np.abs(rollspeeds))
        total_dev_penalty = (yaw_dev + pitch_dev + roll_dev) * ANGULAR_DEVIATION_WEIGHT


        vxs = np.array([v["vx"] for _, v in vel_seq])
        vys = np.array([v["vy"] for _, v in vel_seq])
        vzs = np.array([v["vz"] for _, v in vel_seq])
        yaws = np.array([a["yawspeed"] for _, a in att_seq])
        pitches = np.array([a["pitchspeed"] for _, a in att_seq])
        rolls = np.array([a["rollspeed"] for _, a in att_seq])

        vel_std = np.std(vxs) + np.std(vys) + np.std(vzs)
        att_std = np.std(yaws) + np.std(pitches) + np.std(rolls)
        stability_penalty = (vel_std + att_std) * STABILITY_WEIGHT

        total_reward = tracking_total - stability_penalty - total_dev_penalty
        total_reward = np.clip(total_reward, -CLIP, CLIP)

        return {
            "total": total_reward,
            "vx_score": vx_score,
            "vy_score": vy_score,
            "vz_score": vz_score,
            "yaw_score": yaw_score,
            "pitch_score": pitch_score,
            "roll_score": roll_score,
            "tracking_total": tracking_total,
            "stability_penalty": -stability_penalty,
            "deviation_penalty": -total_dev_penalty,
            "vx_error": vx_errors[-1] if vx_errors else 0.0,
            "vy_error": vy_errors[-1] if vy_errors else 0.0,
            "vz_error": vz_errors[-1] if vz_errors else 0.0,
            "yaw_error": yaw_errors[-1] if yaw_errors else 0.0,
            "pitch_error": pitch_errors[-1] if pitch_errors else 0.0,
            "roll_error": roll_errors[-1] if roll_errors else 0.0
        }


    def is_terminal(self, state):
        return False