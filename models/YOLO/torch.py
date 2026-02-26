import torch
from ultralytics import YOLO

from class_model import Model

from .common import get_torch_dtype


class Model(Model):
    """YOLO inference using Ultralytics weights + Torch (eager)."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._input = None

    def read(self):
        if self.model_name is None:
            raise Exception('Missing --model (e.g. --model yolo11l)')
        weights = str(self.model_name).strip()
        self.model = YOLO(f'{weights}.pt', task='detect')

        if self.device != 'cpu':
            self.model.to(self.device)

        if str(self.precision) == 'fp16':
            self.model.model.half()

    def prepare(self):
        torch_device = self.device if (self.device != 'cpu' and torch.cuda.is_available()) else 'cpu'
        dtype = get_torch_dtype(self.precision) if torch_device == 'cuda' else torch.float32
        self._input = torch.rand(
            (int(self.batch), 3, int(self.imgsz), int(self.imgsz)),
            device=torch_device,
            dtype=dtype,
        )

    def inference(self):
        with torch.no_grad():
            return self.model(self._input, verbose=False)

    def shutdown(self):
        if self.model is not None:
            del self.model
            self.model = None
        self._input = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

