"""Online extension of the shared request CSV must not depend on which worker
extends it, how often, or the global RNG state."""

import os
import shutil
import tempfile
import unittest

import numpy as np

from config import Config
from generator import Generator


def _config(sim_index=0):
    cfg = Config()
    cfg.CONFIG_DEFAULT_CAPACITY = 100
    cfg.SIM_INDEX = sim_index
    return cfg


def _rows(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


class TestDeterministicExtend(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root)

    def _seed_file(self, path, n=500):
        # Pre-existing rows (e.g. an older CSV) are kept as-is.
        with open(path, "w") as f:
            f.write("id,workload,start\n")
            for i in range(n):
                f.write(f"{i},{100.0 + i},{i * 0.01:.5f}\n")

    def test_same_rows_regardless_of_extender(self):
        a_path = os.path.join(self.root, "a.csv")
        b_path = os.path.join(self.root, "b.csv")
        self._seed_file(a_path)
        self._seed_file(b_path)

        # File A: two workers with unrelated global RNG states, many small steps.
        np.random.seed(1)
        g1 = Generator(100000, 50.0, 10.0, _config())
        g2 = Generator(100000, 50.0, 10.0, _config())
        g1.inputRequests(a_path)
        g2.inputRequests(a_path)
        until = 6.0
        for k in range(40):
            np.random.seed(100 + k)
            (g1 if k % 2 else g2).extendUntil(a_path, until)
            until += 20.0
        # File B: one worker, one big extension.
        np.random.seed(999)
        g3 = Generator(100000, 50.0, 10.0, _config())
        g3.inputRequests(b_path)
        g3.extendUntil(b_path, until)

        a, b = _rows(a_path), _rows(b_path)
        n = min(len(a), len(b))
        self.assertGreater(n, 500 + Generator.GEN_CHUNK_ROWS)
        np.testing.assert_array_equal(a[:n], b[:n])
        # Existing rows untouched, ids contiguous, starts non-decreasing.
        np.testing.assert_array_equal(a[:500, 0], np.arange(500))
        np.testing.assert_array_equal(a[:, 0], np.arange(len(a)))
        self.assertTrue(np.all(np.diff(a[:, 2]) >= 0))
        # Both workers hold exactly the file contents.
        for g in (g1, g2):
            got = np.array([(r.getId(), r.getOrgWorkload(), r.getStartTime()) for r in g.reqs])
            np.testing.assert_array_equal(got, a[: len(got)])

    def test_stream_depends_on_sim_index(self):
        paths = []
        for idx in (0, 1):
            p = os.path.join(self.root, f"s{idx}.csv")
            self._seed_file(p, n=10)
            Generator(100000, 50.0, 10.0, _config(idx)).extendUntil(p, 5.0)
            paths.append(p)
        a, b = _rows(paths[0]), _rows(paths[1])
        self.assertFalse(np.array_equal(a[10:20], b[10:20]))


if __name__ == "__main__":
    unittest.main()
