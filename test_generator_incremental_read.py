"""Incremental tail reads of the shared request CSV must load exactly the rows a
full re-read would, including rows appended by another generator."""

import os
import shutil
import tempfile
import unittest

import numpy as np

from config import Config
from generator import Generator


def _config():
    row = [
        "incremental_read_test", "1000", "100000", "req", "102400", "50", "10", "input",
        "0.1", "15", "0.6", "600", "container", "TRUE", "100", "1",
        "30", "30", "100", "1", "-1", "inactive", "verbose", "./result/unused.csv",
    ]
    config = Config()
    config.initialSetup(row, sim_index=0)
    return config


def _snapshot(gen):
    return [(r.getId(), r.getOrgWorkload(), r.getStartTime()) for r in gen.reqs]


def _file_rows(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return [(int(i), float(w), float(s)) for i, w, s in data]


class TestIncrementalRead(unittest.TestCase):
    def test_two_writers_see_full_file(self):
        root = tempfile.mkdtemp()
        try:
            path = os.path.join(root, "reqs.csv")
            cfg = _config()
            np.random.seed(1)
            a = Generator(100000, 50.0, 10.0, cfg)
            b = Generator(100000, 50.0, 10.0, cfg)
            a.createAllRequests(cfg.SIM_LIMIT, 200)
            a.outputRequests(path)
            b.inputRequests(path)
            until = 4.0
            for k in range(30):
                gen = a if k % 3 else b  # uneven interleaving of writers
                gen.extendUntil(path, until, max_new=37)
                other = b if gen is a else a
                other.extendUntil(path, until - 0.5, max_new=37)
                until += 1.3
            a.extendUntil(path, until)
            b.extendUntil(path, until)
            rows = _file_rows(path)
            self.assertEqual(_snapshot(a), rows)
            self.assertEqual(_snapshot(b), rows)
        finally:
            shutil.rmtree(root)

    def test_rewritten_file_is_fully_reread(self):
        root = tempfile.mkdtemp()
        try:
            path = os.path.join(root, "reqs.csv")
            cfg = _config()
            np.random.seed(2)
            g = Generator(100000, 50.0, 10.0, cfg)
            g.createAllRequests(cfg.SIM_LIMIT, 100)
            g.outputRequests(path)
            g.inputRequests(path)
            # Replace the file with a fresh, different one and reload from scratch.
            g2 = Generator(100000, 50.0, 10.0, cfg)
            g2.createAllRequests(cfg.SIM_LIMIT, 300)
            g2.outputRequests(path)
            g.reqs = []
            g.max_loaded_id = -1
            g.inputRequests(path)
            self.assertEqual(_snapshot(g), _file_rows(path))
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
