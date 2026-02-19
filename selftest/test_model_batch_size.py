import re
import subprocess
import os
import pytest
from common import get_combined_output

class TestModelBatchSize:
    """Test cases for batch size handling in test_perf.py."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up: parent dir is perftest, run from there with python.exe."""
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.test_script = os.path.join(self.parent_dir, "test_perf.py")
        self.noisy_model = "selftest.models.noisy_model"

    def run_subprocess(self, args):
        """Run test_perf.py from parent folder using python.exe."""
        cmd = ["python.exe", self.test_script, self.noisy_model] + list(args)
        result = subprocess.run(
            cmd,
            cwd=self.parent_dir,
            capture_output=True,
            text=True,
        )
        return result.stdout, result.stderr, result.returncode

    def check_batch_size_output(self, out, x):
        """
        Verify output contains for each batch size X == 1:
          - '{ "Preparing Batch Size": X },'
          - 'Noisy model.prepare_batch(X)'
          - '{ "Running Batch": X },'
          - '{ "Model Warm Up X" : "<float>" },'
          - '{ "Total Inference X" : "<float>" },'
          - '"Batch Size" : "X",'
        """
        # '{ "Preparing Batch Size": X },'
        assert f'{{ "Preparing Batch Size": {x} }},' in out, (
            f'Expected \'{{ "Preparing Batch Size": {x} }},\' in output'
        )
        # 'Empty model.prepare_batch(X)'
        assert f'{{ "Noisy model" : "prepare_batch({x})" }},' in out, (
            f'Expected {{ "Noisy model" : "prepare_batch({x})" }}, in output'
        )
        # '{ "Running Batch": X },'
        assert f'{{ "Running Batch": {x} }},' in out, (
            f'Expected \'{{ "Running Batch": {x} }},\' in output'
        )
        # '{ "Model Warm Up X" : "<float>" },' (any float)
        warm_up_pattern = re.compile(
            re.escape(f'{{ "Model Warm Up {x}"') + r'\s*:\s*"[^"]+"\s*\},?'
        )
        assert warm_up_pattern.search(out), (
            f'Expected line like \'{{ "Model Warm Up {x}" : "<float>" }},\' in output'
        )
        # '{ "Total Inference X" : "<float>" },'
        total_inf_pattern = re.compile(
            re.escape(f'{{ "Total Inference {x}"') + r'\s*:\s*"[^"]+"\s*\},?'
        )
        assert total_inf_pattern.search(out), (
            f'Expected line like \'{{ "Total Inference {x}" : "<float>" }},\' in output'
        )
        # '"Batch Size" : "X",'
        assert f'"Batch Size" : "{x}",' in out, (
            f'Expected \'"Batch Size" : "{x}",\' in output'
        )

    @pytest.mark.parametrize("batch_sizes", [[1],
    [1, 2],
    [1, 2, 4, 8, 16, 32, 64, 128]])
    def test_batch_size_output_lines(self, batch_sizes):
        """
        Run test_perf.py with selftest.models.noisy_model and batch_sizes --batch-size values.
        """
        # Use few runs per batch to keep test fast
        stdout, stderr, code = self.run_subprocess(["--runs", "2", "--batch-size", ",".join(str(b) for b in batch_sizes)])
        out = get_combined_output(stdout, stderr)
        assert code == 0, f"Subprocess failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"

        for x in batch_sizes:
            self.check_batch_size_output(out, x)

    def test_batch_size_default(self):
        """
        Run test_perf.py with selftest.models.noisy_model and default batch size.
        """
        # Use few runs per batch to keep test fast
        stdout, stderr, code = self.run_subprocess(["--runs", "2"])
        out = get_combined_output(stdout, stderr)
        assert code == 0, f"Subprocess failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"

        for x in [1]:
            self.check_batch_size_output(out, x)
