import time
import cv2
import argparse
from src.video_loader import VideoLoader
from src.detector import Detector
from src.fps_monitor import FPSMonitor
from configs.model_config import MODEL_CONFIG


def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height))


def benchmark(video_path, resolutions=None, device='cpu', max_frames=100):
    if resolutions is None:
        resolutions = [(640, 360), (1280, 720), (1920, 1080)]

    print(f"\n{'='*70}")
    print(f"Benchmark: {video_path}")
    print(f"Device: {device}")
    print(f"{'='*70}")
    print(f"{'Resolution':<15} {'FPS':<10} {'Frames':<10} {'Total Time (s)':<15}")
    print(f"{'-'*70}")

    results = []

    for width, height in resolutions:
        loader = VideoLoader(video_path)
        detector = Detector(
            MODEL_CONFIG['model_path'],
            MODEL_CONFIG['conf_threshold'],
            device
        )
        fps_monitor = FPSMonitor()

        frame_count = 0
        start_time = time.time()

        while frame_count < max_frames:
            ret, frame = loader.read()
            if not ret:
                loader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = loader.read()
                if not ret:
                    break

            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = resize_frame(frame, width, height)

            results_frame = detector.detect(frame)
            fps_monitor.update()
            frame_count += 1

        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0

        print(f"{width}x{height:<8} {avg_fps:<10.2f} {frame_count:<10} {elapsed:<15.2f}")

        results.append({
            'resolution': f'{width}x{height}',
            'fps': avg_fps,
            'frames': frame_count,
            'time': elapsed
        })

        loader.release()

    print(f"{'='*70}")
    print("\nSummary (sorted by FPS):")
    print(f"{'Resolution':<15} {'FPS':<10}")
    for r in sorted(results, key=lambda x: x['fps'], reverse=True):
        print(f"{r['resolution']:<15} {r['fps']:<10.2f}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark video detection performance')
    parser.add_argument('--video', type=str, default='data/videos/input.mp4',
                        help='Path to input video')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda')
    parser.add_argument('--max-frames', type=int, default=50,
                        help='Number of frames to process')
    parser.add_argument('--resolutions', type=str, default='640x360,1280x720,1920x1080',
                        help='Comma-separated resolutions, e.g., 640x360,1280x720')

    args = parser.parse_args()

    resolutions = []
    for res in args.resolutions.split(','):
        w, h = map(int, res.split('x'))
        resolutions.append((w, h))

    benchmark(args.video, resolutions, args.device, args.max_frames)