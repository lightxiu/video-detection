import time


class FPSMonitor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = []
        self.fps = 0.0
        self.frame_count = 0

    def update(self):
        now = time.time()
        self.timestamps.append(now)
        self.frame_count += 1
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

        if len(self.timestamps) >= 2:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            self.fps = (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0

    def get_fps(self):
        return round(self.fps, 1)

    def get_frame_count(self):
        return self.frame_count