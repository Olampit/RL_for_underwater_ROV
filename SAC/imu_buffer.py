# imu_buffer.py
from collections import deque
import threading
import time

class IMUBuffer:
    def __init__(self, max_seconds=1.0, frequency=400):
        self.buffer = deque(maxlen=int(max_seconds * frequency))
        self.lock = threading.Lock()

    def add(self, timestamp, data: dict):
        with self.lock:
            self.buffer.append((timestamp, data))

    def get_all(self):
        with self.lock:
            return list(self.buffer)

    def get_last_n(self, n):
        with self.lock:
            return list(self.buffer)[-n:]
