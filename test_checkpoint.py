"""Checkpoint save/load restores the event-loop position."""

import csv
import os
import shutil
import tempfile
import unittest

from checkpoint import repair_text_tail
from config import Config
from queue_simulator import QueueSimulator


def _tiny_config(root):
    row = [
        "checkpoint_test", "1000", "100000", "req", "102400", "100", "100", "input",
        "0.1", "15", "0.6", "600", "container", "TRUE", "2", "1", "30", "30", "100",
        "1", "-1", "inactive", "verbose", "./result/unused.csv",
    ]
    config = Config()
    config.initialSetup(row, sim_index=0)
    config.CONFIG_DEFAULT_NUM = 2
    config.CONFIG_DEFAULT_START_INSTANCES = 1
    config.OUTPUT_FILE = os.path.join(root, "run_0_result.csv")
    config.OUTPUT_FILE_PACKET = os.path.join(root, "run_0_packet.csv")
    config.OUTPUT_FILE_NUM_INSTANCE = os.path.join(root, "run_0_num_instance.csv")
    config.CONFIG_REQUEST_FILE = os.path.join(root, "requests.csv")
    config.STEADY_ENABLED = False
    config.STEADY_CHECKPOINT_ENABLED = True
    config.STEADY_CHECKPOINT_INTERVAL_SEC = 10**9
    config.STEADY_WARMUP_TIME = 0
    config.STEADY_BATCH_MIN_TIME = 60
    config.STEADY_BATCH_REQS = 1000
    config.SIM_ENABLE_HYBRID_SKIP = True
    config.SIM_ENABLE_SCALE_CHECK_SKIP = True
    return config


class TestCheckpoint(unittest.TestCase):
    def test_repair_drops_torn_row(self):
        root = tempfile.mkdtemp()
        try:
            path = os.path.join(root, "packet.csv")
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "end"])
                writer.writerow([1, 0.1])
            with open(path, "ab") as handle:
                handle.write(b"2,0.2,torn")
            repair_text_tail(path)
            with open(path, newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows, [["id", "end"], ["1", "0.1"]])
        finally:
            shutil.rmtree(root)

    def test_resume_keeps_step_and_completed_count(self):
        root = tempfile.mkdtemp()
        try:
            config = _tiny_config(root)
            sim = QueueSimulator(
                config.SIM_THRESHOLD,
                config.SIM_LIMIT,
                config.SIM_STEP_PER_TIME,
                config.CONFIG_LAMBDA,
                config.CONFIG_MU,
                config,
            )
            with open(config.OUTPUT_FILE_NUM_INSTANCE, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sim_time", "step", "num_hot_instances"])
                sim._num_instance_fh = handle
                before = -1
                for _ in range(50):
                    before = sim.simulateStepWrapper(writer, before)
                sim._num_instance_fh = None
            step = sim.getStep()
            done = sim.num_reqs
            self.assertGreater(step, 0)
            from checkpoint import save_checkpoint

            save_checkpoint(sim, before)

            restored = QueueSimulator(
                config.SIM_THRESHOLD,
                config.SIM_LIMIT,
                config.SIM_STEP_PER_TIME,
                config.CONFIG_LAMBDA,
                config.CONFIG_MU,
                config,
            )
            self.assertTrue(restored._resumed)
            self.assertEqual(restored.getStep(), step)
            self.assertEqual(restored.num_reqs, done)
            self.assertEqual(restored._resume_before_instances, before)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
