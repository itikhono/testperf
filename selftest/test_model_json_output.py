import sys
import os
import subprocess
import pytest
from common import get_combined_output

class TestModelJsonOutput:
    """Run test_perf.py from parent dir with python and assert JSON-parsable output."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Parent dir is perftest (one level up from selftest)."""
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.test_script = os.path.join(self.parent_dir, "test_perf.py")
        self.const_time_model = "selftest.models.const_time_model"

    def run_subprocess(self, args):
        """Run python test_perf.py from parent folder; return (stdout, stderr, returncode)."""
        cmd = [sys.executable, self.test_script] + args
        result = subprocess.run(
            cmd,
            cwd=self.parent_dir,
            capture_output=True,
            text=True,
        )
        return result.stdout, result.stderr, result.returncode

    def test_capture_output_and_read_as_json(self):
        """Capture test_perf.py selftest.models.const_time_model output and read it as JSON."""
        stdout, stderr, code = self.run_subprocess([self.const_time_model, '--delay_all', '0.01'])
        assert code == 0, stdout + stderr
        get_combined_output(stdout, stderr) # should not raise an exception
