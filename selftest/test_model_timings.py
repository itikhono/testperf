import json
import os
import subprocess
import pytest
from common import get_combined_output

import sys
# Add parent directory to sys.path to import models from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestModelTimings:
    """Run test_perf.py from parent dir with python.exe; parse and prepare timing data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Parent dir is perftest (one level up from selftest)."""
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.test_script = os.path.join(self.parent_dir, "test_perf.py")
        self.const_time_model = "selftest.models.const_time_model"
        self.accuracy = 0.15

    def run_subprocess(self, args):
        """Run python.exe test_perf.py from parent folder; return (stdout, stderr, returncode)."""
        cmd = ["python.exe", self.test_script] + args
        result = subprocess.run(
            cmd,
            cwd=self.parent_dir,
            capture_output=True,
            text=True,
        )
        return result.stdout, result.stderr, result.returncode
    
    def test_const_time_model_behavior(self):
        import models.const_time_model
        model = models.const_time_model.Model()
        for key in model.delays:
            assert model.delays[key] == 0.1, f"Expected model.delays[{key}] to be 0.1, got {model.delays[key]}"
        model.set_delays(['--delay_all', '2'])
        for key in model.delays:
            assert model.delays[key] == 2, f"Expected model.delays[{key}] to be 0.1, got {model.delays[key]}"
        idx = 1
        for key in model.delays:
            model.set_delays(['--delay-'+key, str(idx)])
            assert model.delays[key] == idx, f"Expected model.delays[{key}] to be {idx}, got {model.delays[key]}"
            idx += 1

    def test_increment_time_model_behavior(self):
        import models.increment_time_model
        model = models.increment_time_model.Model()
        assert model.increment == 0.01, f"Expected model.increment to be 0.01, got {model.increment}"
        assert model.delays == model.default_delays, f"Expected model.delays to be {model.default_delays}, got {model.delays}"
        for key in model.delays:
            assert model.delays[key] == 0.1, f"Expected model.delays[{key}] to be 0.1, got {model.delays[key]}"

        model.set_delays(['--delay_all', '2'])
        assert model.delays == model.default_delays, f"Expected model.delays to be {model.default_delays}, got {model.delays}"
        for key in model.delays:
            assert model.delays[key] == 2, f"Expected model.delays[{key}] to be 0.1, got {model.delays[key]}"
        assert model.delays == model.default_delays, f"Expected model.delays to be {model.default_delays}, got {model.delays}"

        idx = 1
        for key in model.delays:
            model.set_delays(['--delay-'+key, str(idx)])
            assert model.delays[key] == idx, f"Expected model.delays[{key}] to be {idx}, got {model.delays[key]}"
            idx += 1

    def check_model_timings(self, args):
        """
        Run test_perf.py with selftest/models/const_time_model.py, parse output as JSON,
        and prepare structures for checking timing data (Steps, Read Summary, Inference Summary, etc.).
        """
        stdout, stderr, code = self.run_subprocess([self.const_time_model] + args)
        assert code == 0, get_combined_output(stdout, stderr)

        out = get_combined_output(stdout, stderr)
        import json
        data = json.loads(out)

        import models.const_time_model
        model = models.const_time_model.Model()
        model.set_delays(args)

        # Looking for a Steps section in the output
        steps = data.get("Steps", None)
        assert steps is not None, f"Steps not found in output: {out}"

        # Looking for a "Preparing Batch Size" step
        prepare_count = 0
        for step in steps:
            if isinstance(step, dict) and "Preparing Batch Size" in step:
                prepare_count += 1
        assert prepare_count > 0, f"Expected > 0 \"Preparing Batch Size\" steps, got {prepare_count}"
        total_preparing_batches = None
        for step in steps:
            if isinstance(step, dict) and "Total Preparing Batches" in step:
                total_preparing_batches = float(step["Total Preparing Batches"])
                break
        # Looking for a "Total Preparing Batches" step
        assert total_preparing_batches is not None, f"Total Preparing Batches not found in output: {out}"
        diff = abs(total_preparing_batches - prepare_count * model.delays['prepare-batch'])
        assert diff < self.accuracy, f"Expected total preparing batches to be close to prepare count * prepare batch delay, got {total_preparing_batches} and {prepare_count * model.delays['prepare-batch']}, diff {diff}"

        # Looking for a "Model 1st Read" step
        model_1st_read = None
        for step in steps:
            if isinstance(step, dict) and "Model 1st Read" in step:
                model_1st_read = float(step["Model 1st Read"])
                break
        assert model_1st_read is not None, f"Model 1st Read not found in output: {out}"
        assert abs(model_1st_read - model.delays['read1st']) < self.accuracy, f"Expected model 1st read to be close to read1st delay, got {model_1st_read} and {model.delays['read1st']}, diff {diff}"

        # Looking for a "Read Runs" step
        read_runs = None
        for step in steps:
            if isinstance(step, dict) and "Read Runs" in step:
                read_runs = int(step["Read Runs"])
                break
        assert read_runs is not None, f"Read Runs not found in output: {out}"
        assert read_runs > 0, f"Expected > 0 Read Runs, got {read_runs}"
        # Looking for a "Total Read" step
        total_read = None
        for step in steps:
            if isinstance(step, dict) and "Total Read" in step:
                total_read = float(step["Total Read"])
                break
        assert total_read is not None, f"Total Read not found in output: {out}"
        diff = abs(total_read - read_runs * model.delays['read'])
        assert diff < self.accuracy, f"Expected total read to be close to read runs * read delay, got {total_read} and {read_runs * model.delays['read']}, diff {diff}"
        
        # Looking for a "Read Times" step
        read_times = None
        for step in steps:
            if isinstance(step, dict) and "Read Times" in step:
                read_times = step["Read Times"]
                break
        assert read_times is not None, f"Read Times not found in output: {out}"
        assert len(read_times) == read_runs, f"Expected {read_runs} Read Times, got {len(read_times)}"
        times = []
        sum = 0
        for time in read_times:
            times.append(float(time["Time"]))
            diff = abs(float(time["Time"]) - model.delays['read'])
            assert diff < self.accuracy, f"Expected read time to be close to read delay, got {float(time["Time"])} and {model.delays['read']}, diff {diff}"
            sum += float(time["Time"])
        diff = abs(sum - total_read)
        assert diff < self.accuracy, f"Expected total read to be close to sum of read times, got {sum} and {total_read}, diff {diff}"

        # Looking for a "Read Summary" step
        read_summary = None
        for step in steps:
            if isinstance(step, dict) and "Read Summary" in step:
                read_summary = step["Read Summary"]
                break
        assert read_summary is not None, f"Read Summary not found in output: {out}"
        assert len(read_summary) == 3, f"Expected 3 items in Read Summary, got {len(read_summary)}"
        diff = abs(float(read_summary['Minimum']) - min(times))
        assert diff < self.accuracy, f"Expected minimum read time to be close to minimum of read times, got {read_summary['Minimum']} and {min(times)}, diff {diff}"
        diff = abs(float(read_summary['Maximum']) - max(times))
        assert diff < self.accuracy, f"Expected maximum read time to be close to maximum of read times, got {read_summary['Maximum']} and {max(times)}, diff {diff}"
        diff = abs(float(read_summary['Average']) - sum / len(times))
        assert diff < self.accuracy, f"Expected average read time to be close to average of read times, got {read_summary['Average']} and {sum / len(times)}, diff {diff}"

    def test_const_time_model_timings(self):
        self.check_model_timings([])

    def test_increment_time_model_timings(self):
        self.check_model_timings(["--delay_all", "0.01", "--increment", "0.01"])
