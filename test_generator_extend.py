import os
import tempfile
import unittest

import numpy as np

from config import Config
from generator import Generator
from limit import Limit


class TestGeneratorExtendNoRewind(unittest.TestCase):
    def _cfg(self):
        cfg = Config()
        # Minimal fields used by Generator workload sampling.
        cfg.CONFIG_DEFAULT_CAPACITY = 100
        return cfg

    def test_stale_next_arrival_does_not_rewind_shared_csv(self):
        cfg = self._cfg()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reqs.csv")
            gen_a = Generator(100000, 1600.0, 100.0, cfg)
            gen_a.seed(1)
            gen_a.generate_and_write(path, Limit.LIMIT_TIME, 10.0)

            # Simulate another worker extending the file further in time.
            gen_b = Generator(100000, 1600.0, 100.0, cfg)
            gen_b.seed(2)
            appended = gen_b.ensure_file_until(path, 20.0)
            self.assertGreater(appended, 0)

            # Worker A still has a stale next_arrival_time near its old horizon.
            peek_id, peek_start = gen_a._peek_last_row(path)
            self.assertGreater(peek_start, 10.0)
            gen_a.next_arrival_time = 5.0  # stale / behind disk
            gen_a.last_start_time = 9.0
            gen_a.max_loaded_id = 0  # force reload of newer rows

            with open(path) as fh:
                before_rows = sum(1 for _ in fh) - 1
            gen_a.extendUntil(path, 25.0, max_new=5000)
            with open(path) as fh:
                after_rows = sum(1 for _ in fh) - 1
            self.assertGreaterEqual(after_rows, before_rows)

            data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            starts = data[:, 2]
            self.assertTrue(np.all(np.diff(starts) >= -1e-12), "request starts must stay non-decreasing")
            self.assertGreaterEqual(float(starts[-1]), float(peek_start))


if __name__ == "__main__":
    unittest.main()
