import os

import migraphx
import numpy as np

from class_model import Model

from .common import get_np_dtype, onnx_name, try_export_model


class Model(Model):
    """YOLO inference using direct MIGraphX with cache"""

    def __init__(self):
        super().__init__()
        self.mx_model = None

    def prepare_batch(self, batch):
        if self.model_name is None:
            raise Exception('Missing --model (e.g. --model yolo11l)')
        file_path = self.get_file_path(onnx_name(self.model_name, batch, self.precision, self.imgsz))
        try_export_model(file_path, self.model_name, batch, self.precision, self.imgsz, dynamic=False)
        cache_path = file_path[:-4] + 'mxr'

        if not os.path.exists(cache_path):
            try:
                model = migraphx.parse_onnx(file_path)
                model.compile(migraphx.get_target('gpu'))
                migraphx.save(model, cache_path)
                del model
            except Exception as e:
                raise Exception(f'Failed to compile and save MIGraphX model: {e}')

    def read(self):
        file_path = self.get_file_path(onnx_name(self.model_name, self.batch, self.precision, self.imgsz))
        cache_path = file_path[:-4] + 'mxr'
        self.mx_model = migraphx.load(cache_path)

    def prepare(self):
        dtype = get_np_dtype(self.precision)
        self.input_data = np.random.randn(self.batch, 3, self.imgsz, self.imgsz).astype(dtype)

    def inference(self):
        return self.mx_model.run({'images': self.input_data})

    def shutdown(self):
        if self.mx_model:
            del self.mx_model
            self.mx_model = None
