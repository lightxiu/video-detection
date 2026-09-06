import cv2
class VideoLoader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def release(self):
        self.cap.release()