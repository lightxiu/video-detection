# CV 视频目标检测

基于 YOLOv8 的模块化视频目标检测 Pipeline。

## 项目架构

```
video_detection_demo/
├── configs/
│   └── model_config.py      # 模型配置
├── src/
│   ├── __init__.py
│   ├── video_loader.py       # 视频加载模块
│   ├── detector.py           # 目标检测模块
│   ├── visualizer.py         # 可视化标注模块
│   └── video_writer.py       # 视频输出模块
├── data/videos/              # 输入视频目录
├── output/results/            # 输出视频目录
├── main.py                   # 主程序
└── requirements.txt          # 依赖
```

## 模块职责

### 1. VideoLoader (video_loader.py)

**职责：** 视频读取

**核心类：**
```python
class VideoLoader:
    def __init__(self, video_path)
    def read() -> Tuple[bool, np.ndarray]  # 返回 (是否成功, 帧)
    def release()
```

**技术要点：**
- 使用 OpenCV `cv2.VideoCapture` 读取视频
- `read()` 返回 BGR 格式图像
- 资源用完必须 `release()`

---

### 2. Detector (detector.py)

**职责：** 目标检测

**核心类：**
```python
class Detector:
    def __init__(self, model_path, conf=0.5, device='cuda')
    def detect(frame) -> List[Results]  # YOLOv8 检测结果
```

**技术要点：**
- 使用 ultralytics YOLOv8 模型
- 支持 CPU/GPU 切换
- 返回 `Results` 对象，包含 `boxes.xyxy`、`boxes.conf`、`boxes.cls`

---

### 3. Visualizer (visualizer.py)

**职责：** 检测结果可视化

**核心类：**
```python
class Visualizer:
    def __init__(self, class_names)
    def draw(frame, results) -> np.ndarray  # 返回带标注的帧
```

**技术要点：**
- 使用 OpenCV 绘制边界框和标签
- 每帧显示检测统计信息（左下角）
- 不同类别使用不同颜色

---

### 4. VideoWriter (video_writer.py)

**职责：** 视频保存

**核心类：**
```python
class VideoWriter:
    def __init__(self, output_path, fps, width, height)
    def write(frame)
    def release()
```

**技术要点：**
- 使用 OpenCV `cv2.VideoWriter`
- FourCC 编码格式：`mp4v`
- 尺寸必须与输入帧匹配

---

## 配置说明

`configs/model_config.py`:

```python
MODEL_CONFIG = {
    'model_path': 'yolov8n.pt',      # 模型路径
    'conf_threshold': 0.5,            # 置信度阈值
    'iou_threshold': 0.45,             # IOU 阈值
    'device': 'cuda'                   # 'cuda' 或 'cpu'
}
```

---

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行

```bash
python main.py
```

### 3. 输出

- 输入：`data/videos/input.mp4`
- 输出：`output/results/output.mp4`

---

## 数据流

```
输入视频 → VideoLoader → 原始帧
    ↓
Detector → 检测结果(results)
    ↓
Visualizer → 带标注帧
    ↓
VideoWriter → 输出视频
```

---

## 性能对比

| 设备 | FPS | 适用场景 |
|------|-----|----------|
| GPU (cuda) | ~30+ | 实时处理 |
| CPU | ~5-10 | 离线处理 |

---

## 可扩展性设计

### 架构层次

```
┌─────────────────────────────────────────────────┐
│                  Detector                       │
│  ┌──────────────────────────────────────────┐   │
│  │        DetectorBackend (抽象接口)          │   │
│  └──────────────────────────────────────────┘   │
│         ↑              ↑              ↑         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ PyTorch  │   │ TensorRT │   │   ONNX   │     │
│  │ Backend  │   │ Backend  │   │ Backend  │     │
│  └──────────┘   └──────────┘   └──────────┘     │
└─────────────────────────────────────────────────┘
```

### 扩展能力

| 层次 | 说明 | 示例 |
|------|------|------|
| **模块级** | 每个模块独立，可单独替换 | VideoLoader → CameraLoader |
| **后端级** | 抽象接口，支持多推理引擎 | PyTorch → TensorRT |
| **配置级** | 配置文件统一管理，无需改代码 | `'backend': 'tensorrt'` |

### 添加新后端

```python
class ONNXBackend(DetectorBackend):
    def __init__(self, onnx_path, conf=0.5):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path)
        self.conf = conf

    def detect(self, frame):
        input_tensor = self._preprocess(frame)
        outputs = self.session.run(None, {'images': input_tensor})
        return self._postprocess(outputs)

    def get_names(self):
        return {0: 'person', 1: 'car', ...}
```

### 切换后端

```python
# configs/model_config.py
MODEL_CONFIG = {
    'backend': 'pytorch',  # 'pytorch' / 'tensorrt' / 'onnx'
    ...
}
```

---

## 运行截图

**主程序运行效果**

<img width="768" height="432" alt="P2026-04-11_15-53-17" src="https://github.com/user-attachments/assets/c65423a3-5e08-450b-b834-bffe974f78cf" />




**不同分辨率测试**

<img width="381" height="195" alt="image" src="https://github.com/user-attachments/assets/7eec1bf8-3840-4f31-baf8-d8d9ea7b5fee" />



