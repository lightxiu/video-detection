# CV Pipeline 项目交接文档

## 项目目标
构建一个完整的视频目标检测Pipeline项目，用于简历展示。

**核心技能点：**
- 视频流处理
- Pipeline架构设计
- OpenCV使用

## 技术栈
- OpenCV (opencv-python)
- YOLOv8 (ultralytics)
- PyTorch

## 项目目录结构
```
video_detection_demo/
├── configs/
│   └── model_config.py      # 模型配置（置信度阈值、模型路径等）
├── models/                   # 模型权重目录
├── src/
│   ├── __init__.py
│   ├── video_loader.py      # 视频读取模块
│   ├── detector.py           # 目标检测模块
│   ├── visualizer.py         # 可视化模块（画框+标签）
│   └── video_writer.py       # 视频输出模块
├── data/
│   └── videos/               # 输入视频目录
├── output/
│   └── results/              # 输出视频目录
├── main.py                   # 主程序（Pipeline串联）
├── requirements.txt
└── README.md
```

## 已完成状态
- [x] 目录结构创建
- [x] requirements.txt 创建
- [ ] 虚拟环境搭建和依赖安装（用户计划在服务器上用现成环境）

## 待完成任务

### 1. 环境准备
```bash
# 创建并激活环境
conda create -n video_detection python=3.10
conda activate video_detection

# 安装依赖
pip install opencv-python>=4.8.0 ultralytics>=8.0.0 torch torchvision numpy
```

### 2. 获取YOLOv8预训练模型
使用ultralytics库自动下载：
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  #会自动下载
```

### 3. 实现各模块（按顺序）

#### 3.1 configs/model_config.py
```python
MODEL_CONFIG = {
    'model_path': 'yolov8n.pt',
    'conf_threshold': 0.5,
    'iou_threshold': 0.45,
    'device': 'cuda'  # 或 'cpu'
}
```

#### 3.2 src/video_loader.py
使用OpenCV读取视频：
```python
import cv2

class VideoLoader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()
```

#### 3.3 src/detector.py
使用YOLOv8进行推理：
```python
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path, conf=0.5, device='cuda'):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf)
        return results
```

#### 3.4 src/visualizer.py
在帧上画检测框和标签：
```python
import cv2

class Visualizer:
    def __init__(self, class_names):
        self.class_names = class_names

    def draw(self, frame, results):
        # 遍历results，绘制边界框和标签
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f"{self.class_names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        return frame
```

#### 3.5 src/video_writer.py
使用OpenCV保存视频：
```python
import cv2

class VideoWriter:
    def __init__(self, output_path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
```

#### 3.6 main.py（Pipeline串联）
```python
from src.video_loader import VideoLoader
from src.detector import Detector
from src.visualizer import Visualizer
from src.video_writer import VideoWriter
from configs.model_config import MODEL_CONFIG

def main():
    video_path = 'data/videos/input.mp4'
    output_path = 'output/results/output.mp4'

    loader = VideoLoader(video_path)
    detector = Detector(MODEL_CONFIG['model_path'],
                       MODEL_CONFIG['conf_threshold'],
                       MODEL_CONFIG['device'])
    visualizer = Visualizer(detector.model.names)
    writer = None

    while True:
        ret, frame = loader.read()
        if not ret:
            break

        results = detector.detect(frame)
        frame = visualizer.draw(frame, results)

        if writer is None:
            fps = 30
            writer = VideoWriter(output_path, fps, frame.shape[1], frame.shape[0])
        writer.write(frame)

    loader.release()
    writer.release()

if __name__ == '__main__':
    main()
```

### 4. 准备测试视频
在 `data/videos/` 目录放置测试视频（如道路监控视频）

### 5. 运行Demo
```bash
python main.py
```

## 注意事项
1. YOLOv8会自动下载预训练权重到用户目录
2. GPU版本需要根据CUDA版本选择对应的torch安装命令
3. OpenCV的VideoWriter需要正确设置fourcc编码格式

## 项目亮点（在简历中强调）
1. **模块化Pipeline设计**：解耦的视频读取、检测、可视化、输出模块
2. **OpenCV熟练使用**：视频I/O、图像绘制
3. **端到端处理能力**：从视频输入到带标注视频输出
