import cv2
import numpy as np


class Visualizer:
    def __init__(self, class_names):
        self.class_names = class_names
        self.colors = self._generate_colors(len(class_names))

    def _generate_colors(self, n):
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(n, 3), dtype=np.uint8)
        return [tuple(map(int, color)) for color in colors]

    def _get_color(self, cls):
        if cls < len(self.colors):
            return self.colors[cls]
        return (0, 255, 0)

    def draw(self, frame, results, fps=None):
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{self.class_names[cls]} {conf:.2f}"
                color = self._get_color(cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        stats = self._count_objects(results)
        frame = self._draw_stats(frame, stats)

        if fps is not None:
            frame = self._draw_fps(frame, fps)

        return frame

    def _count_objects(self, results):
        counts = {}
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                name = self.class_names[cls]
                counts[name] = counts.get(name, 0) + 1
        return counts

    def _draw_stats(self, frame, stats):
        h, w = frame.shape[:2]
        x, y = 10, h - 10
        line_height = 20
        bg_color = (50, 50, 50)
        text_color = (255, 255, 255)

        cv2.rectangle(frame, (5, h - 30 - len(stats) * line_height),
                     (200, h), bg_color, -1)

        for i, (name, count) in enumerate(stats.items()):
            text = f"{name}: {count}"
            y_pos = y - i * line_height
            cv2.putText(frame, text, (x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        return frame

    def _draw_fps(self, frame, fps):
        text = f"FPS: {fps}"
        cv2.putText(frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame