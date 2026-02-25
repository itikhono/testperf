import os


class Model:
    """
    Base class / interface for inference backends used by `test_perf.py`.

    The harness imports a backend module (e.g. `models.YOLO.ort`) and instantiates
    `Model()`. Then it calls the hooks below in a fixed order while measuring time.

    **Call order (simplified)**

    - For each `batch` from `--batch-size`:
      - `prepare_batch(batch)`  # export/compile/cache for that batch
    - `read1st()`                # first model load (timed separately)
    - many times: `readnth()`    # repeated loads to sample "read time"
    - For each `batch` again (main benchmark):
      - `shutdown()` (if switching batch)
      - `read()`
      - `warm_up()`              # default: prepare() + inference1st()
      - `prepare()`
      - 10x `inference()`        # untimed
      - timed loop:
        - `reset_inference_run()`
        - while `next_inference_run()`:
          - `inference()`
      - (next batch)
    - `shutdown()` (final)

    **Fields set by the harness**

    - `self.batch`: current batch size for the main benchmark loop
    - `self.model_name`: weights/model identifier from `--model` (no extension)
    - `self.imgsz`: image size (e.g. 640)
    - `self.precision`: `"fp16"` or `"fp32"`

    **Important rule**

    - Do not print to stdout from backend code (stdout is reserved for JSON).
      If you need logs, print to stderr.

    **Minimal backend example**

    ```python
    import numpy as np
    import onnxruntime as ort
    from class_model import Model

    class Model(Model):
        def __init__(self):
            super().__init__()
            self.sess = None

        def prepare_batch(self, batch):
            # export/cache files for this batch (optional)
            pass

        def read(self):
            # load model into memory for current self.batch
            self.sess = ort.InferenceSession("model.onnx")

        def prepare(self):
            # allocate inputs for current self.batch
            self.input = np.random.rand(self.batch, 3, 640, 640).astype(np.float32)

        def inference(self):
            # run one inference
            self.sess.run([], {"images": self.input})

        def shutdown(self):
            self.sess = None
    ```
    """

    def __init__(self):
        """Initialize default harness fields; backends may add extra state."""
        self.batch = 1
        self.total_inference_runs = 100
        self.current_inference_run = 0
        # Optional configuration populated by test_perf.py for models that need it.
        self.model_name = None
        self.imgsz = 640
        self.precision = 'fp32'  # 'fp16' or 'fp32'
        pass

    def prepare_batch(self, batch):
        """Prepare artifacts for a specific batch size (export/compile/cache)."""
        pass

    def reset_inference_run(self):
        """Reset internal run counter used by `next_inference_run()`."""
        self.current_inference_run = 0

    def next_inference_run(self):
        """Return True while the harness should keep running timed `inference()`."""
        if self.current_inference_run >= self.total_inference_runs:
            return False
        self.current_inference_run += 1
        return True

    def read(self):
        """Load the model for the current `self.batch` into memory."""
        pass

    def read1st(self):
        """First read hook (timed separately); default delegates to `read()`."""
        self.read()

    def readnth(self):
        """Repeated read hook; default delegates to `read()`."""
        self.read()

    def warm_up(self):
        """Warm up to avoid counting one-time initialization in timings."""
        self.prepare()
        self.inference1st()

    def prepare(self):
        """Prepare inputs for the current `self.batch`."""
        pass

    def inference(self):
        """Run one inference for the current prepared inputs."""
        pass

    def inference1st(self):
        """First inference hook; default delegates to `inference()`."""
        self.inference()

    def inferencenth(self):
        """Repeated inference hook; default delegates to `inference()`."""
        self.inference()

    def shutdown(self):
        """Release resources; called when switching batch sizes and at the end."""
        pass

    def get_file_path(self, file_name):
        """Return an absolute path inside repo-local `temp/` (for caches/artifacts)."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp', file_name)

    def __str__(self):
        """Return backend description (used in reports/debug)."""
        return self.__doc__
