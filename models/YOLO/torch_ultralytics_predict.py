import torch
from ultralytics import YOLO

from class_model import Model

from .common import get_torch_dtype


class Model(Model):
    """YOLO inference with using Ultralytics predict() + Torch.Compile"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.compile_mode = 'max-autotune-no-cudagraphs'
        self.source = None

    def read(self):
        if self.model_name is None:
            raise Exception('Missing --model (e.g. --model yolo11l)')
        weights = str(self.model_name).strip()
        self.model = YOLO(f'{weights}.pt', task='detect')

    def prepare(self):
        torch_device = 'cuda' if (torch.cuda.is_available() and self.device != 'cpu') else 'cpu'
        dtype = get_torch_dtype(self.precision) if torch_device == 'cuda' else torch.float32
        self.source = torch.rand(
            (self.batch, 3, self.imgsz, self.imgsz),
            device=torch_device,
            dtype=dtype,
        )

    def inference(self):
        half = str(self.precision) == 'fp16'
        return self.model.predict(
            source=self.source,
            imgsz=self.imgsz,
            device=self.device,
            half=half,
            batch=self.batch,
            compile=self.compile_mode,
            save=False,
            verbose=False,
        )

    def shutdown(self):
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
