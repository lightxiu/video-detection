import cv2


class VideoWriter:
    def __init__(self, output_path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise ValueError(f"Cannot open video writer: {output_path}")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()