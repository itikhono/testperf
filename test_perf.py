import argparse
import os
import platform
import sys
from time import perf_counter


class JsonWriter:
    """Print JSON lines to stdout."""

    def __init__(self) -> None:
        self._out = sys.stdout

    def line(self, s: str) -> None:
        print(s, file=self._out)


class Timer:
    """Collect perf_counter checkpoints and print spent time for a category."""

    def __init__(self, w: JsonWriter) -> None:
        self._w = w
        self.times: list[float] = []

    def checkpoint(self, reset: bool = True) -> None:
        if reset:
            self.times = []
        self.times.append(perf_counter())

    def spent(self, category: str) -> float:
        self.times.append(perf_counter())
        spent_time = self.times[-1] - self.times[0]
        self._w.line(f'{{ "{category}" : "{spent_time}" }},')
        return spent_time


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('backend', nargs='?', default='test_model')
    ap.add_argument('--batch-size', dest='batch', default=None)
    ap.add_argument('--runs', dest='runs', default=None)
    ap.add_argument('--only-prepare', action='store_true', default=False)
    ap.add_argument('--model', dest='model', default=None)
    ap.add_argument('--imgsz', dest='imgsz', default=None)
    ap.add_argument('--precision', dest='precision', choices=['fp16', 'fp32'], default='fp16')
    args, _unknown = ap.parse_known_args(argv)
    return args


def normalize_backend_name(name: str) -> str:
    # Accept module name with dot delimiter or directory separator.
    mod = str(name).replace('/', '.').replace('\\', '.')
    return mod[:-3] if mod.endswith('.py') else mod


def parse_batches(arg: str | None, w: JsonWriter) -> list[int]:
    batches = [1]
    if arg is None:
        return batches
    try:
        return [int(x) for x in str(arg).split(',')]
    except Exception as e:
        w.line(f'{{ "Error": "Failed to set batch size {e}, using default [{", ".join(str(b) for b in batches)}]" }},')
        return batches


def load_backend_module(name: str, w: JsonWriter) -> object:
    try:
        return __import__(name, fromlist=['Model'])
    except Exception as e:
        w.line(f'{{ "Error": "Failed to load model {e}" }},')

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        path_parts = name.split('.')
        prefix = ''
        for part in path_parts:
            last_dir = str(cur_dir)
            cur_dir = os.path.join(cur_dir, part)
            if not os.path.exists(cur_dir):
                print('\n\nAvailable options: ', file=sys.stderr)
                print(
                    '\n'.join([f'{prefix[1:]}.{x}...' for x in os.listdir(last_dir)]),
                    file=sys.stderr,
                )
                break
            prefix = prefix + '.' + part
        raise SystemExit(1)


def summarize_from_checkpoints(checkpoints: list[float]) -> tuple[float, float, float]:
    # Match the existing logic: segments are (t[i] - t[i-1]) for i in [1..len-2].
    last = checkpoints[0]
    max_time = checkpoints[1] - last
    min_time = checkpoints[1] - last
    for t in checkpoints[1:-1]:
        dt = t - last
        last = t
        max_time = max(dt, max_time)
        min_time = min(dt, min_time)
    avg_time = (checkpoints[-1] - checkpoints[0]) / (len(checkpoints) - 2)
    return min_time, max_time, avg_time


def durations_from_checkpoints(checkpoints: list[float]) -> list[float]:
    return [checkpoints[i] - checkpoints[i - 1] for i in range(1, len(checkpoints) - 1)]


def main() -> None:
    args = parse_args(sys.argv[1:])
    w = JsonWriter()

    backend_name = normalize_backend_name(args.backend)

    w.line(f'{{ "Model": "{backend_name}",')
    w.line(f'"Hostname": "{platform.node()}",')
    w.line(f'"Platform": "{platform.system()} {platform.release()} {platform.version()}",')
    w.line('"Steps": [')

    script_run_time = perf_counter()
    test_model = load_backend_module(backend_name, w)

    if not os.path.exists('./temp'):
        os.makedirs('./temp')

    batches = parse_batches(args.batch, w)
    t = Timer(w)

    try:
        model = test_model.Model()
    except Exception as e:
        w.line(f'{{ "Error": "Failed to create model {e}" }},')
        raise SystemExit(2)

    if args.model is not None:
        model.model_name = str(args.model)
    if args.imgsz is not None:
        try:
            model.imgsz = int(args.imgsz)
        except Exception:
            pass
    if args.precision is not None:
        model.precision = str(args.precision)

    if args.runs is not None:
        try:
            model.total_inference_runs = int(args.runs)
        except Exception as e:
            w.line(f'{{ "Error": "Failed to set runs {e}, using default {model.total_inference_runs} runs" }},')

    # Prepare artifacts for all requested batch sizes.
    t.checkpoint()
    for batch in batches:
        w.line(f'{{ "Preparing Batch Size": {batch} }},')
        model.prepare_batch(batch)
    t.spent('Total Preparing Batches')

    if args.only_prepare:
        t.checkpoint()
        model.shutdown()
        t.spent('Model Shutdown')
        w.line('{ "Status": "Done" }')
        w.line('] }')
        return

    # Read timing: 1st + repeated reads.
    t.checkpoint()
    model.read1st()
    t.spent('Model 1st Read')

    first_read_time = t.times[-1] - t.times[0]
    if first_read_time < 60:
        read_runs = min(50, (600 // int(first_read_time if first_read_time > 1 else 1)))
        w.line(f'{{ "Read Runs": {read_runs} }},')
        t.checkpoint()
        for _ in range(read_runs):
            model.readnth()
            t.checkpoint(reset=False)
        t.spent('Total Read')
    else:
        w.line(f'{{ "Error": "Read time is too long {t.times[-1] - t.times[0]}, skipping read tests" }},')
        t.times.append(t.times[-1])  # Fake time to avoid index error

    read_times = durations_from_checkpoints(t.times)
    min_time, max_time, avg_time = summarize_from_checkpoints(t.times)

    w.line('{ "Read Times": [')
    for item in read_times[:-1]:
        w.line(f'{{ "Time" : "{item}" }},')
    w.line(f'{{ "Time" : "{read_times[-1]}" }}')
    w.line('] },')

    w.line('{ "Read Summary": {')
    w.line(f'"Minimum" : "{min_time}",')
    w.line(f'"Maximum" : "{max_time}",')
    w.line(f'"Average" : "{avg_time}"')
    w.line('} },')

    read_times.append({'Minimum': min_time, 'Maximum': max_time, 'Average': avg_time})

    # Main benchmark loop.
    model.batch = None
    inference_times: dict[int, list[float | dict[str, float]]] = {}
    warm_up_times: dict[int, float] = {}

    for batch in batches:
        w.line(f'{{ "Running Batch": {batch} }},')
        if model.batch is None or model.batch != batch:
            model.shutdown()
            model.batch = batch

        t.checkpoint()
        model.read()
        model.warm_up()
        warm_up_times[batch] = t.spent(f'Model Warm Up {batch}')

        model.reset_inference_run()
        model.prepare()

        # Few empty runs (untimed).
        for _ in range(10):
            model.inference()

        # Timed loop.
        t.checkpoint()
        while model.next_inference_run():
            model.inference()
            t.checkpoint(reset=False)
        t.spent(f'Total Inference {batch}')

        inference_times[batch] = durations_from_checkpoints(t.times)
        min_time, max_time, avg_time = summarize_from_checkpoints(t.times)

        w.line('{ "Inference Times": [')
        for item in inference_times[batch][:-1]:
            w.line(f'{{ "Time" : "{item}" }},')
        w.line(f'{{ "Time" : "{inference_times[batch][-1]}" }}')
        w.line('] },')

        inference_times[batch].append({'Minimum': min_time, 'Maximum': max_time, 'Average': avg_time})

        w.line('{ "Inference Summary": {')
        w.line(f'"Batch Size" : {batch},')
        w.line(f'"Minimum" : "{min_time}",')
        w.line(f'"Maximum" : "{max_time}",')
        w.line(f'"Average" : "{avg_time}"')
        w.line('} },')

    # Shutdown + report.
    t.checkpoint()
    model.shutdown()
    t.spent('Model Shutdown')

    total_time = perf_counter() - script_run_time
    w.line(f'{{ "Total Time": {total_time} }},')

    try:
        import reports

        reports.performance_report(
            model,
            backend_name,
            args.model,
            args.precision,
            read_times,
            inference_times,
            warm_up_times,
            batches,
        )
    except Exception as e:
        w.line(f'{{ "Error": "Failed to generate XLS report {e}" }},')

    w.line('{ "Status": "Done" }')
    w.line('] }')


if __name__ == '__main__':
    main()
