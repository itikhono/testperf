import os

import numpy as np
import torch
from ultralytics import YOLO

from class_model import Model

from .common import onnx_name, try_export_model


class Model(Model):
    """YOLO inference with using Ultralytics predict() + ONNXRuntime"""

    def __init__(self):
        super().__init__()
        self.model = None
        # Default to CPU for compatibility: ORT GPU EP selection inside Ultralytics can try to load CUDA EP.
        self.device = 'cpu'
        self.source = None

    def prepare_batch(self, batch):
        if self.model_name is None:
            raise Exception('Missing --model (e.g. --model yolo11l)')
        file_path = self.get_file_path(onnx_name(self.model_name, batch, self.precision, self.imgsz))
        try_export_model(file_path, self.model_name, batch, self.precision, self.imgsz, dynamic=False)

    def read(self):
        file_path = self.get_file_path(onnx_name(self.model_name, self.batch, self.precision, self.imgsz))
        if not os.path.exists(file_path):
            raise Exception(f'Model file {file_path} not found')
        self.model = YOLO(file_path, task='detect')

    def prepare(self):
        img = (np.random.rand(self.imgsz, self.imgsz, 3) * 255).astype(np.uint8)
        self.source = [img] * self.batch

    def inference(self):
        half = str(self.precision) == 'fp16'
        return self.model.predict(source=self.source,
                                  imgsz=self.imgsz,
                                  device=self.device,
                                  half=half,
                                  batch=self.batch,
                                  save=False,
                                  verbose=False)

    def shutdown(self):
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
