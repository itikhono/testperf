import os


def get_np_dtype(precision):
    import numpy as np

    return {
        'fp16': np.float16,
        'fp32': np.float32,
    }.get(str(precision), np.float32)


def get_torch_dtype(precision):
    import torch

    return {
        'fp16': torch.float16,
        'fp32': torch.float32,
    }.get(str(precision), torch.float32)


def get_ort_input_np_dtype(sess):
    import numpy as np

    t = sess.get_inputs()[0].type if sess and sess.get_inputs() else None
    return {
        'tensor(float16)': np.float16,
        'tensor(float)': np.float32,
    }.get(t, np.float32)


def get_ort_input_torch_dtype(sess):
    import torch

    t = sess.get_inputs()[0].type if sess and sess.get_inputs() else None
    return {
        'tensor(float16)': torch.float16,
        'tensor(float)': torch.float32,
    }.get(t, torch.float32)


def onnx_name(model_name, batch, precision, imgsz):
    name = str(model_name).strip()
    pfx = f'{name}_fp16' if str(precision) == 'fp16' else name
    return f'{pfx}_{int(batch)}b_{int(imgsz)}.onnx'


def try_export_model(file_path, model_name, batch, precision, imgsz, dynamic=False):
    if os.path.exists(file_path):
        return
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    from ultralytics import YOLO

    half = str(precision) == 'fp16'
    weights = str(model_name).strip()
    model = YOLO(f'{weights}.pt', task='detect')
    onnx_path = model.export(format='onnx', imgsz=int(imgsz), batch=int(batch), half=bool(half), dynamic=bool(dynamic))

    if str(onnx_path) != str(file_path):
        os.replace(str(onnx_path), file_path)
