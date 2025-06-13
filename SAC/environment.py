#environment.py

from pymavlink import mavutil
import numpy as np
import time
import subprocess
from joystick_input import FakeJoystick
import math
import random

from imu_reader import attitude_buffer, velocity_buffer


SERVO_MIN = 1100
SERVO_MAX = 1900
SERVO_IDLE = 1500



def input_to_pwm(value):
    """
    Converts a normalized thrust value (-1 to 1) to a servo PWM value.

    Parameters:
        value (float): Thrust value between -1.0 and 1.0.

    Returns:
        int: PWM signal between SERVO_MIN (1100) and SERVO_MAX (1900), idle at 1500.

    Called in:
        ROVEnvironment.apply_action().
    """
    if abs(value) < 0.05:
        return SERVO_IDLE
    pwm = SERVO_IDLE + (value * 400)
    return int(max(SERVO_MIN, min(SERVO_MAX, pwm)))

class ROVEnvironment:
    """
    Environment for controlling a ROV, collecting IMU/velocity data, and computing RL rewards.
    """
    def __init__(self, action_map, connection, latest_imu):
        """
        Initialize the environment with action mapping, MAVLink connection, and joystick.

        Parameters:
            action_map (List[dict]): List of thrust configurations for 8 motors.
            connection: MAVLink connection object.
            latest_imu: Not used (reserved).

        Called in:
            train script.
        """
        self.action_map = action_map
        self.connection = connection
        self.latest_imu = latest_imu
        self.joystick = FakeJoystick()        
        self.target_velocity = self.joystick.get_target()

    def apply_action(self, action_idx):
        """
        Sends motor commands based on the selected action index.

        Parameters:
            action_idx (int): Index in the action map.

        Called in:
            I think never, since we have a better "duplicate" of this function in the env wrapper. 
        """
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
        print(f"[ACTION] Sent: {action}")

    def get_state(self):
        """
        Computes and returns current state from attitude and velocity buffers.

        Returns:
            dict: Dictionary containing means and variances of velocities and angular rates.

        Called in:
            reset(), compute_reward(), state_to_index().
        """
        state = {}

        att_seq = attitude_buffer.get_all()
        
        if att_seq:
            # Extract arrays for yawspeed, pitchspeed, rollspeed only if key exists
            yaws = np.array([abs(d["yawspeed"]) for _, d in att_seq if "yawspeed" in d])
            pitchs = np.array([abs(d["pitchspeed"]) for _, d in att_seq if "pitchspeed" in d])
            rolls = np.array([abs(d["rollspeed"]) for _, d in att_seq if "rollspeed" in d])

            if yaws.size > 0:
                state["yaw_var"] = yaws.var()
                state["yaw_mean"] = yaws.mean()
            if pitchs.size > 0:
                state["pitch_var"] = pitchs.var()
                state["pitch_mean"] = pitchs.mean()
            if rolls.size > 0:
                state["roll_var"] = rolls.var()
                state["roll_mean"] = rolls.mean()


        vel_seq = velocity_buffer.get_all()
        
        if vel_seq:
            # Build structured numpy arrays for velocity components
            vxs = np.array([v["vx"] for _, v in vel_seq])
            vys = np.array([v["vy"] for _, v in vel_seq])
            vzs = np.array([v["vz"] for _, v in vel_seq])
            mags = np.array([v["mag"] for _, v in vel_seq])

            state["vx_mean"] = vxs.mean()
            state["vy_mean"] = vys.mean()
            state["vz_mean"] = vzs.mean()

            state["vx_var"] = vxs.var()
            state["vy_var"] = vys.var()
            state["vz_var"] = vzs.var()

            state["vel_mag_avg"] = mags.mean()
            state["vel_mag_var"] = mags.var()


        return state

    def random_orientation_quat(self, max_angle_deg=15):
        """
        Generates a small random orientation as a quaternion.

        Parameters:
            max_angle_deg (float): Maximum angle deviation in degrees.

        Returns:
            dict: Quaternion with keys x, y, z, w.

        Called in:
            reset().
        """
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
        """
        Respawns the robot at a random pose and resets joystick goal.

        Returns:
            dict: Initial state after reset.

        Called in:
            RL training loop, before every episode.
        """
        # Random small variation in spawn position 
        px = round(random.uniform(-1.0, 1.0), 2)
        py = round(random.uniform(4.0, 6), 2)
        pz = round(random.uniform(9.5, 10.5), 2)
        
        

        # Random orientation
        quat = self.random_orientation_quat(max_angle_deg=15)
        qx, qy, qz, qw = quat["x"], quat["y"], quat["z"], quat["w"]

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

        subprocess.run(cmd)
        
        self.joystick.next_episode()

        return self.get_state()



    def stop_motors(self, connection):
        """
        Sends 1500 PWM (idle) to all motors to stop them.

        Parameters:
            connection: MAVLink connection object.

        Called in:
            Whenever we do a reset, this function is called (in train)
        """
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
            
            
    



    def compute_reward(self, state):
        """
        Computes the reward based on how well the observed mean and std match the target,
        using smooth Gaussian-shaped rewards and a gradual bonus.

        Returns:
            dict: Contains reward components (total, shaped terms, errors, rates).
        """
        goal = self.joystick.get_target()

        total_vel_score = 0.0
        total_std_score = 0.0

        def gaussian_score(x, target, sigma=0.05):
            """Reward is highest when x == target, falls off smoothly."""
            return np.exp(-((x - target)**2) / (2 * sigma**2))

        def smooth_bonus(score, scale, threshold=0.85):
            """
            Returns a bonus scaled between 0 and `scale` if `score` > threshold.
            """
            if score < threshold:
                return 0.0
            return scale * (score - threshold) / (1.0 - threshold)
        # Linear velocities
        for axis in ["vx", "vy", "vz"]:
            target_mean = goal[axis]["mean"]
            target_std = goal[axis]["std"]

            observed_mean = state.get(f"{axis}_mean", 0.0)
            observed_var = state.get(f"{axis}_var", 0.0)
            observed_std = np.sqrt(observed_var)

            mean_score = gaussian_score(observed_mean, target_mean, sigma=0.1)
            std_score = gaussian_score(observed_std, target_std, sigma=0.05)

            total_vel_score += mean_score
            total_std_score += std_score

        # Angular rates
        angular_means = {
            "yaw_rate": abs(state.get("yaw_mean", 0.0)),
            "pitch_rate": abs(state.get("pitch_mean", 0.0)),
            "roll_rate": abs(state.get("roll_mean", 0.0)),
        }

        angular_vars = {
            "yaw_rate": state.get("yaw_var", 0.0),
            "pitch_rate": state.get("pitch_var", 0.0),
            "roll_rate": state.get("roll_var", 0.0),
        }

        for axis in ["yaw_rate", "pitch_rate", "roll_rate"]:
            target_mean = goal[axis]["mean"]
            target_std = goal[axis]["std"]

            observed_mean = angular_means[axis]
            observed_std = np.sqrt(angular_vars[axis])

            mean_score = gaussian_score(observed_mean, target_mean, sigma=0.005)
            std_score = gaussian_score(observed_std, target_std, sigma=0.002)

            total_vel_score += mean_score
            total_std_score += std_score

        # Bonus: gradually increases when both scores are close to ideal (6.0 max)
        vel_bonus = smooth_bonus(total_vel_score / 6.0, scale=3.0)
        std_bonus = smooth_bonus(total_std_score / 6.0, scale=2.0)
        bonus = vel_bonus + std_bonus


        # Final shaped reward
        total_reward = 3.0 * total_vel_score + 5.0 * total_std_score + bonus

        return {
            "total": total_reward,
            "velocity_score": total_vel_score,
            "std_score": total_std_score,
            "bonus": bonus,
            "yaw_rate": angular_means["yaw_rate"],
            "pitch_rate": angular_means["pitch_rate"],
            "roll_rate": angular_means["roll_rate"],
        }









    def state_to_index(self, state):
        """
        Converts a state into a discretized tuple index based on key rounding.

        Parameters:
            state (dict): Current state.

        Returns:
            tuple: Rounded values of 15 selected state features.

        Called in:
            used for tabular Q-learning or state mapping.
        """
        keys = [
            "yaw_var", "yaw_mean",
            "pitch_var", "pitch_mean",
            "roll_var", "roll_mean",
            "vx_mean", "vy_mean", "vz_mean",
            "vx_var", "vy_var", "vz_var",
            "vel_mag_avg", "vel_mag_var"
        ]
        
        return tuple(round(state.get(k, 0.0), 3) for k in keys)





    def is_terminal(self, state):
        """
        Determines if the current state is terminal (always False here).

        Parameters:
            state (dict): Current state.

        Returns:
            bool: False.

        Called in:
            RL loop condition.
        """
        return False

