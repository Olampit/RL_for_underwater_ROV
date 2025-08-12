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

        """
        with self.lock:
            return list(self.buffer)[-n:]


    def get_since(self, since_time, max_age=None):
        now = time.time()
        with self.lock:
            results = [
                (t, d) for t, d in self.buffer
                if t >= since_time and (max_age is None or t >= now - max_age)
            ]

            if len(results) >= 2:
                return results
            elif len(results) == 1:
                # Append latest if it's not already in results
                latest = self.buffer[-1]
                if results[0] != latest:
                    return [results[0], latest]
                elif len(self.buffer) >= 2:
                    return [self.buffer[-2], self.buffer[-1]]
                else:
                    return results
            else:
                # No matches — fallback to last two from buffer if available
                if len(self.buffer) >= 2:
                    return [self.buffer[-2], self.buffer[-1]]
                elif self.buffer:
                    return [self.buffer[-1]]
                else:
                    return []



