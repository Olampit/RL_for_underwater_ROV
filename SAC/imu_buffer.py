# imu_buffer.py
from collections import deque
import threading
import time

class IMUBuffer:
    """
    Circular thread-safe buffer to store timestamped sensor data.
    """
    def __init__(self, max_seconds=1.0, frequency=400):
        """
        Initialize the IMU buffer with a fixed time window.

        Parameters:
            max_seconds (float): Maximum duration to keep in buffer.
            frequency (int): Expected frequency (Hz) of incoming data.

        Called in:
            imu_reader.py: used to create attitude_buffer and velocity_buffer.
        """
        self.buffer = deque(maxlen=int(max_seconds * frequency))
        self.lock = threading.Lock()

    def add(self, timestamp, data: dict):
        """
        Add a timestamped dictionary of IMU data to the buffer.

        Parameters:
            timestamp (float): Time when the data was recorded.
            data (dict): Dictionary containing IMU or velocity measurements.

        Called in:
            imu_reader.py > start_imu_listener and OdomListener.odom_callback.
        """
        with self.lock:
            self.buffer.append((timestamp, data))

    def get_all(self):
        """
        Retrieve all buffered data as a list.

        Returns:
            List[Tuple[float, dict]]: All (timestamp, data) tuples in the buffer.

        Called in:
            environment.py > get_state().
        """
        with self.lock:
            return list(self.buffer)

    def get_last_n(self, n):
        """
        Get the last n entries from the buffer.

        Parameters:
            n (int): Number of recent elements to retrieve.

        Returns:
            List[Tuple[float, dict]]: The last n (timestamp, data) entries.

        Not called usually (but still there)
        """
        with self.lock:
            return list(self.buffer)[-n:]
