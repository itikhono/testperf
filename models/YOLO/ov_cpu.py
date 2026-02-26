import numpy as np
import openvino as ov

from class_model import Model

from .common import get_np_dtype, onnx_name, try_export_model


class Model(Model):
    """YOLO inference with using OpenVINO"""

    def __init__(self):
        super().__init__()
        self.core = ov.Core()
        self.ov_model = None
        self.compiled_model = None

    def prepare_batch(self, batch):
        if self.model_name is None:
            raise Exception('Missing --model (e.g. --model yolo11l)')
        file_path = self.get_file_path(onnx_name(self.model_name, batch, self.precision, self.imgsz))
        try_export_model(file_path, self.model_name, batch, self.precision, self.imgsz, dynamic=False)

    def read(self):
        file_path = self.get_file_path(onnx_name(self.model_name, self.batch, self.precision, self.imgsz))
        self.ov_model = self.core.read_model(file_path)
        self.compiled_model = self.core.compile_model(self.ov_model, 'CPU')

    def prepare(self):
        dtype = get_np_dtype(self.precision)
        self.input_data = {
            'images': np.random.randn(self.batch, 3, self.imgsz, self.imgsz).astype(dtype),
        }

    def inference(self):
        return self.compiled_model(self.input_data)

    def shutdown(self):
        pass
