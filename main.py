from src.video_loader import VideoLoader
from src.detector import Detector
from src.visualizer import Visualizer
from src.video_writer import VideoWriter
from src.fps_monitor import FPSMonitor
from configs.model_config import MODEL_CONFIG


def main():
    video_path = 'data/videos/input.mp4'
    output_path = 'output/results/output_frames.mp4'

    loader = VideoLoader(video_path)
    detector = Detector(
        MODEL_CONFIG['model_path'],
        MODEL_CONFIG['conf_threshold'],
        MODEL_CONFIG['device'],
        MODEL_CONFIG.get('backend', 'pytorch')
    )
    visualizer = Visualizer(detector.names)
    fps_monitor = FPSMonitor()
    writer = None

    while True:
        ret, frame = loader.read()
        if not ret:
            break

        results = detector.detect(frame)
        fps_monitor.update()
        annotated_frame = visualizer.draw(frame, results, fps_monitor.get_fps())

        if writer is None:
            writer = VideoWriter(output_path, 30, frame.shape[1], frame.shape[0])
        writer.write(annotated_frame)

    loader.release()
    writer.release()
    print(f"Output saved to: {output_path}")
    print(f"Total frames: {fps_monitor.get_frame_count()}")


if __name__ == '__main__':
    main()