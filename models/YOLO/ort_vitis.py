import numpy as np
import onnxruntime as ort

from class_model import Model

from .common import get_ort_input_np_dtype, onnx_name, try_export_model


class Model(Model):
    """YOLO inference with using Vitis AI Execution Provider"""

    def __init__(self):
        super().__init__()
        self.sess = None
        self.sess_data = {'providers': ['VitisAIExecutionProvider']}
        if self.sess_data['providers'][0] not in ort.get_available_providers():
            raise Exception('Vitis AI Execution Provider is not available')

    def prepare_batch(self, batch):
        if self.model_name is None:
            raise Exception('Missing --model (e.g. --model yolo11l)')
        file_path = self.get_file_path(onnx_name(self.model_name, batch, self.precision, self.imgsz))
        try_export_model(file_path, self.model_name, batch, self.precision, self.imgsz, dynamic=False)

    def read(self):
        file_path = self.get_file_path(onnx_name(self.model_name, self.batch, self.precision, self.imgsz))
        self.sess = ort.InferenceSession(file_path, **self.sess_data)

    def prepare(self):
        dtype = get_ort_input_np_dtype(self.sess)
        self.input_data = {
            'images': np.random.randn(self.batch, 3, self.imgsz, self.imgsz).astype(dtype),
        }

    def inference(self):
        return self.sess.run([], input_feed=self.input_data)

    def shutdown(self):
        pass
