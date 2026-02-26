import torch
from ultralytics import YOLO

from class_model import Model

from .common import get_torch_dtype


class Model(Model):
    """YOLO inference using Ultralytics weights + Torch.Compile."""

    def __init__(self):
        super().__init__()
        if not torch.cuda.is_available():
            raise Exception('CUDA is not available')
        self.model = None
        self.device = 'cuda'
        self.compile_mode = 'max-autotune-no-cudagraphs'
        self._input = None

    def read(self):
        if self.model_name is None:
            raise Exception('Missing --model (e.g. --model yolo11l)')
        weights = str(self.model_name).strip()
        self.model = YOLO(f'{weights}.pt', task='detect')
        self.model.to(self.device)

        if str(self.precision) == 'fp16':
            self.model.model.half()

        # Compile the underlying nn.Module for speed.
        self.model.model = torch.compile(self.model.model, mode=self.compile_mode)

    def prepare(self):
        dtype = get_torch_dtype(self.precision)
        self._input = torch.rand(
            (int(self.batch), 3, int(self.imgsz), int(self.imgsz)),
            device=self.device,
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

