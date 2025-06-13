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
        Computes the reward based on the velocity and angular rate errors relative to the target.

        Parameters:
            state (dict): Current state including velocity and rate statistics.

        Returns:
            dict: Dictionary with reward components: total, progress_reward, stability, bonus, errors.

        Called in:
            When doing a step. 
        """
        goal = self.joystick.get_target()

        total_vel_error = 0.0
        total_std_error = 0.0

        # Linear velocities
        for axis in ["vx", "vy", "vz"]:
            mean_target = goal[axis]["mean"]
            std_target = goal[axis]["std"]

            mean = state.get(f"{axis}_mean", 0.0)
            var = state.get(f"{axis}_var", 0.0)
            std = np.sqrt(var)

            total_vel_error += abs(mean - mean_target)
            total_std_error += abs(std - std_target)

        # Angular rates
        yaw_mean = abs(state.get("yaw_mean", 0.0))
        pitch_mean = abs(state.get("pitch_mean", 0.0))
        roll_mean = abs(state.get("roll_mean", 0.0))

        yaw_var = state.get("yaw_var", 0.0)
        pitch_var = state.get("pitch_var", 0.0)
        roll_var = state.get("roll_var", 0.0)

        for axis, mean, var in [
            ("yaw_rate", yaw_mean, yaw_var),
            ("pitch_rate", pitch_mean, pitch_var),
            ("roll_rate", roll_mean, roll_var),
        ]:
            mean_target = goal[axis]["mean"]
            std_target = goal[axis]["std"]
            std = np.sqrt(var)

            total_vel_error += abs(mean - mean_target)
            total_std_error += abs(std - std_target)

        # Final reward terms
        progress_reward = -total_vel_error
        stability_reward = -total_std_error

        bonus = 5.0 if total_vel_error < 0.05 and total_std_error < 0.05 else 0.0

        total = 3.0 * progress_reward + 5.0 * stability_reward + bonus

        return {
            "total": total,
            "progress_reward": progress_reward,
            "stability": stability_reward,  # same key as before
            "bonus": bonus,
            "vel_error": total_vel_error,
            "yaw_rate": yaw_mean,
            "pitch_rate": pitch_mean,
            "roll_rate": roll_mean,
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

