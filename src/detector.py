from src.backends import create_backend


class Detector:
    def __init__(self, model_path, conf=0.5, device='cuda', backend='pytorch'):
        self.backend = create_backend(
            backend_type=backend,
            model_path=model_path,
            conf=conf,
            device=device
        )

    def detect(self, frame):
        return self.backend.detect(frame)

    @property
    def model(self):
        return self.backend.model if hasattr(self.backend, 'model') else None

    @property
    def names(self):
        return self.backend.get_names()