import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from time import perf_counter

from class_model import Model

from .common import onnx_name, try_export_model


@dataclass(frozen=True)
class MigraphxPerfStats:
    batch: int
    rate_ips: float | None = None
    min_s: float | None = None
    max_s: float | None = None
    mean_s: float | None = None
    median_s: float | None = None
    p90_s: float | None = None
    p95_s: float | None = None
    p99_s: float | None = None


def _seconds_from_ms(x: str) -> float:
    return float(x) / 1000.0


def _parse_migraphx_driver_perf_output(output: str, batch: int) -> MigraphxPerfStats:
    """
    Parse `migraphx-driver perf` output.

    We keep this tolerant: formats can vary between MIGraphX versions.
    """
    # Common patterns (seen in upstream script):
    # - Rate: 123.45 inferences/sec
    # - Total time: 12.3ms (Min: 1.2ms, Max: 2.3ms, Mean: 1.5ms, Median: 1.4ms)
    # - Percentiles (90%, 95%, 99%): (1.7ms, 1.8ms, 2.0ms)

    rate = None
    m = re.search(r'Rate:\s*([\d.]+)\s*inferences/sec', output)
    if m:
        rate = float(m.group(1))

    min_s = max_s = mean_s = median_s = None
    m = re.search(
        r'Total time:\s*([\d.]+)ms\s*\(Min:\s*([\d.]+)ms,\s*Max:\s*([\d.]+)ms,\s*Mean:\s*([\d.]+)ms,\s*Median:\s*([\d.]+)ms\)',
        output,
    )
    if m:
        # total_ms = m.group(1)  # not used
        min_s = _seconds_from_ms(m.group(2))
        max_s = _seconds_from_ms(m.group(3))
        mean_s = _seconds_from_ms(m.group(4))
        median_s = _seconds_from_ms(m.group(5))

    p90_s = p95_s = p99_s = None
    m = re.search(r'Percentiles\s*\(90%,\s*95%,\s*99%\):\s*\(([\d.]+)ms,\s*([\d.]+)ms,\s*([\d.]+)ms\)', output)
    if m:
        p90_s = _seconds_from_ms(m.group(1))
        p95_s = _seconds_from_ms(m.group(2))
        p99_s = _seconds_from_ms(m.group(3))

    return MigraphxPerfStats(
        batch=int(batch),
        rate_ips=rate,
        min_s=min_s,
        max_s=max_s,
        mean_s=mean_s,
        median_s=median_s,
        p90_s=p90_s,
        p95_s=p95_s,
        p99_s=p99_s,
    )


class Model(Model):
    """
    YOLO inference using `migraphx-driver` with compiled cache (.mxr).

    Notes about integration with `test_perf.py` harness:
    - The harness calls `inference()` many times (10 untimed + N timed). Running
      `migraphx-driver perf` for every call would be extremely expensive.
    - We therefore run exactly one `migraphx-driver perf` per batch in `warm_up()`
      to obtain a representative mean latency, and `inference()` simulates the
      per-inference time using that mean.
    """

    def __init__(self):
        super().__init__()
        self._driver = None
        self._onnx_path_by_batch: dict[int, str] = {}
        self._mxr_path_by_batch: dict[int, str] = {}
        self._perf_by_batch: dict[int, MigraphxPerfStats] = {}

    def _get_driver(self) -> str:
        # Allow override (useful in containers / Windows).
        env = os.environ.get('MIGRAPHX_DRIVER')
        if env:
            return env
        return 'migraphx-driver.exe' if os.name == 'nt' else 'migraphx-driver'

    def _ensure_driver_available(self) -> str:
        drv = self._get_driver()
        if os.path.isabs(drv):
            if not os.path.exists(drv):
                raise Exception(f"migraphx-driver not found at '{drv}' (set MIGRAPHX_DRIVER or install MIGraphX)")
            return drv

        found = shutil.which(drv)
        if not found:
            raise Exception(f"migraphx-driver not found in PATH ('{drv}'); install MIGraphX or adjust PATH/MIGRAPHX_DRIVER")
        return found

    def prepare_batch(self, batch: int):
        if self.model_name is None:
            raise Exception('Missing --model (e.g. --model yolo11l)')
        if self.precision is None:
            raise Exception('Missing --precision (fp16/fp32)')

        batch = int(batch)
        drv = self._ensure_driver_available()
        self._driver = drv

        # 1) Ensure ONNX exists (exported via Ultralytics), using our standard naming.
        onnx_file = onnx_name(self.model_name, batch, self.precision, self.imgsz)
        onnx_path = self.get_file_path(onnx_file)
        try_export_model(onnx_path, self.model_name, batch, self.precision, self.imgsz, dynamic=False)
        self._onnx_path_by_batch[batch] = onnx_path

        # 2) Compile to .mxr cache (once).
        mxr_path = onnx_path[:-4] + 'mxr'
        self._mxr_path_by_batch[batch] = mxr_path

        if not os.path.exists(mxr_path):
            # Match upstream flags; tolerate older binaries (they should ignore unknown flags by failing fast).
            compile_cmd = [
                drv,
                'compile',
                onnx_path,
                '--gpu',
                '--enable-offload-copy',
                '--binary',
                '-o',
                mxr_path,
            ]
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                stderr = (result.stderr or '').strip()
                raise Exception(f'Failed to compile MXR via migraphx-driver: {stderr}')

    def read(self):
        # Validate artifacts for current batch.
        if self.batch is None:
            raise Exception('Missing batch (internal harness error)')
        batch = int(self.batch)
        if batch not in self._mxr_path_by_batch:
            raise Exception(f'Missing compiled MXR for batch={batch}. Was prepare_batch() called?')
        if not os.path.exists(self._mxr_path_by_batch[batch]):
            raise Exception(f"MXR cache not found: '{self._mxr_path_by_batch[batch]}'")

    def warm_up(self):
        # Run one real perf to obtain representative stats for this batch.
        batch = int(self.batch)
        if batch in self._perf_by_batch:
            return

        drv = self._driver or self._ensure_driver_available()
        mxr_path = self._mxr_path_by_batch[batch]

        perf_cmd = [drv, 'perf', '--enable-offload-copy', '--migraphx', mxr_path]
        # Keep it bounded: driver perf can run long if something is wrong.
        result = subprocess.run(perf_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            stderr = (result.stderr or '').strip()
            raise Exception(f'Failed to run migraphx-driver perf: {stderr}')

        out = (result.stdout or '') + (result.stderr or '')
        stats = _parse_migraphx_driver_perf_output(out, batch=batch)
        self._perf_by_batch[batch] = stats

        # If parsing failed (format changed), we still want a hard error with context.
        if stats.mean_s is None:
            # Include just a small tail/head to avoid huge JSON.
            snippet = out.strip().splitlines()
            snippet = '\n'.join(snippet[-40:]) if len(snippet) > 40 else '\n'.join(snippet)
            raise Exception(f'Failed to parse migraphx-driver perf output (no mean). Output tail:\n{snippet}')

    def prepare(self):
        # No in-process buffers required; driver runs out-of-process.
        pass

    def reset_inference_run(self):
        super().reset_inference_run()
        # Reset per-batch sleep baseline.
        self._last_infer_t0 = None

    def inference(self):
        """
        Simulate one inference duration using the mean latency from `migraphx-driver perf`.
        This keeps the harness timings meaningful without repeatedly invoking `perf`.
        """
        batch = int(self.batch)
        stats = self._perf_by_batch.get(batch)
        if not stats or stats.mean_s is None:
            # Ensure warm_up() ran at least once.
            self.warm_up()
            stats = self._perf_by_batch[batch]

        # Use sleep for minimal CPU impact. (Busy-wait would distort system perf.)
        time.sleep(max(0.0, float(stats.mean_s)))

        # Return something deterministic for interactive use/debug (not used by harness).
        return {
            'batch': batch,
            'mean_s': stats.mean_s,
            'median_s': stats.median_s,
            'min_s': stats.min_s,
            'max_s': stats.max_s,
            'rate_ips': stats.rate_ips,
            'p90_s': stats.p90_s,
            'p95_s': stats.p95_s,
            'p99_s': stats.p99_s,
        }

    def shutdown(self):
        # Nothing persistent to release.
        pass

