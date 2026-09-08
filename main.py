import argparse
import os
import time

from src.video_loader import VideoLoader
from src.detector import Detector
from src.visualizer import Visualizer
from src.video_writer import VideoWriter
from src.fps_monitor import FPSMonitor
from src.report_generator import DetectionStats, ReportGenerator
from configs.model_config import MODEL_CONFIG, LLM_CONFIG, CLASS_NAME_MAP


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 video detection + LLM report')
    parser.add_argument('--video', default='data/videos/input.mp4',
                        help='input video path')
    parser.add_argument('--no-report', action='store_true',
                        help='skip LLM report generation')
    args = parser.parse_args()

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_path = f'output/results/{video_name}_annotated.mp4'
    report_path = f'output/reports/{video_name}_report.md'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    loader = VideoLoader(args.video)
    detector = Detector(
        MODEL_CONFIG['model_path'],
        MODEL_CONFIG['conf_threshold'],
        MODEL_CONFIG['device'],
        MODEL_CONFIG.get('backend', 'pytorch')
    )
    visualizer = Visualizer(detector.names)
    fps_monitor = FPSMonitor()
    stats = DetectionStats(source=args.video, fps=loader.get_fps())
    reporter = ReportGenerator(
        base_url=LLM_CONFIG['base_url'],
        model=LLM_CONFIG['model'],
        timeout=LLM_CONFIG['timeout'],
        class_name_map=CLASS_NAME_MAP,
    ) if (LLM_CONFIG['enabled'] and not args.no_report) else None

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
                path = f'{os.path.splitext(report_path)[0]}_{time.strftime("%H%M%S")}.md'
                reporter.save(reporter.generate(stats.summary()), path)
                print(f"[report] {path}")
                last_report_at = now

    loader.release()
    writer.release()
    print(f"Output saved to: {output_path}")
    print(f"Total frames: {fps_monitor.get_frame_count()}")

    if reporter is not None:
        report = reporter.generate(stats.summary())
        reporter.save(report, report_path)
        print(f"Report saved to: {report_path}")
        print('-' * 60)
        print(report)


if __name__ == '__main__':
    main()
