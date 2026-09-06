import os
import time

from src.video_loader import VideoLoader
from src.detector import Detector
from src.visualizer import Visualizer
from src.video_writer import VideoWriter
from src.fps_monitor import FPSMonitor
from src.report_generator import DetectionStats, ReportGenerator
from configs.model_config import MODEL_CONFIG, LLM_CONFIG, CLASS_NAME_MAP


def _report_path(output_dir, suffix=''):
    name = 'inspection_report'
    if suffix:
        name = f'{name}_{suffix}'
    return os.path.join(output_dir, f'{name}.md')


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
    stats = DetectionStats(source=video_path, fps=loader.get_fps())
    reporter = ReportGenerator(
        base_url=LLM_CONFIG['base_url'],
        model=LLM_CONFIG['model'],
        timeout=LLM_CONFIG['timeout'],
        class_name_map=CLASS_NAME_MAP,
    ) if LLM_CONFIG['enabled'] else None

    writer = None
    report_interval = LLM_CONFIG.get('report_interval', 0)
    last_report_at = time.time()

    while True:
        ret, frame = loader.read()
        if not ret:
            break

        results = detector.detect(frame)
        stats.update(results, detector.names)
        fps_monitor.update()
        annotated_frame = visualizer.draw(frame, results, fps_monitor.get_fps())

        if writer is None:
            writer = VideoWriter(output_path, 30, frame.shape[1], frame.shape[0])
        writer.write(annotated_frame)

        if reporter is not None and report_interval > 0:
            now = time.time()
            if now - last_report_at >= report_interval:
                path = _report_path(LLM_CONFIG['output_dir'], time.strftime('%H%M%S'))
                reporter.save(reporter.generate(stats.summary()), path)
                print(f"[report] {path}")
                last_report_at = now

    loader.release()
    writer.release()
    print(f"Output saved to: {output_path}")
    print(f"Total frames: {fps_monitor.get_frame_count()}")

    if reporter is not None:
        path = _report_path(LLM_CONFIG['output_dir'])
        report = reporter.generate(stats.summary())
        reporter.save(report, path)
        print(f"Report saved to: {path}")
        print('-' * 60)
        print(report)


if __name__ == '__main__':
    main()
