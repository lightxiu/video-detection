from abc import ABC, abstractmethod


class DetectorBackend(ABC):
    @abstractmethod
    def detect(self, frame):
        pass

    @abstractmethod
    def get_names(self):
        pass


class PyTorchBackend(DetectorBackend):
    def __init__(self, model_path, conf=0.5, device='cuda'):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self, frame):
        return self.model(frame, conf=self.conf, device=self.device)

    def get_names(self):
        return self.model.names


class TensorRTBackend(DetectorBackend):
    def __init__(self, engine_path, conf=0.5):
        raise NotImplementedError("TensorRT backend not yet implemented")

    def detect(self, frame):
        raise NotImplementedError("TensorRT backend not yet implemented")

    def get_names(self):
        raise NotImplementedError("TensorRT backend not yet implemented")


def create_backend(backend_type='pytorch', **kwargs):
    backends = {
        'pytorch': PyTorchBackend,
        'tensorrt': TensorRTBackend,
    }
    if backend_type not in backends:
        raise ValueError(f"Unknown backend: {backend_type}. Available: {list(backends.keys())}")
    return backends[backend_type](**kwargs)