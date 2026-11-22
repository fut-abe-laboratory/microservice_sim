"""SETUP/SHUTDOWN instances are timer-driven: they are not stepped every step,
yet they finish setup at exactly the same step as the old per-step countdown."""

import os
import shutil
import tempfile
import unittest

from config import Config
from queue_simulator import QueueSimulator
from status import Status


def _config(root, mode, setup_sec):
    row = [
        "timer_heap_test", "1000", "100000", "req", "102400", "50", "10", "input",
        "0.1", "15", "0.6", "600", mode, "TRUE", "1000", "0",
        str(setup_sec), "1", "100", "1", "-1", "inactive", "verbose", "./result/unused.csv",
    ]
    config = Config()
    config.initialSetup(row, sim_index=0)
    config.OUTPUT_FILE = os.path.join(root, "run_0_result.csv")
    config.OUTPUT_FILE_PACKET = os.path.join(root, "run_0_packet.csv")
    config.OUTPUT_FILE_NUM_INSTANCE = os.path.join(root, "run_0_num_instance.csv")
    config.CONFIG_REQUEST_FILE = os.path.join(root, "requests.csv")
    config.STEADY_ENABLED = False
    config.STEADY_CHECKPOINT_ENABLED = False
    return config


def _sim(config):
    return QueueSimulator(config.SIM_THRESHOLD, config.SIM_LIMIT, config.SIM_STEP_PER_TIME,
                          config.CONFIG_LAMBDA, config.CONFIG_MU, config)


class TestSetupTimerHeap(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_many_setups_not_stepped_and_finish_on_time(self):
        sim = _sim(_config(self.root, "serverless", 2))
        scaler = sim.getCluster().getScaler()
        insts = [scaler.coldStartInstance() for _ in range(500)]
        self.assertTrue(all(i.getStatus() == Status.SETUP for i in insts))
        setup_steps = insts[0].setuptimer  # S * step_per_time + 1
        steps = 0
        while insts[0].getStatus() == Status.SETUP:
            due = scaler.instances_needing_step(sim.getStep())
            # Only instances whose setup ends this step are stepped.
            self.assertEqual(len(due), 500 if steps == setup_steps - 1 else 0)
            for inst in due:
                inst.runStep(sim.getStep())
            sim.incrementStep()
            steps += 1
            self.assertLess(steps, setup_steps + 5)
        # Old semantics: decrement then flip at timer <= 0, i.e. after `setup_steps` steps.
        self.assertEqual(steps, setup_steps)
        self.assertTrue(all(i.getStatus() == Status.ACTIVE for i in insts))

    def test_skip_advances_virtual_timer(self):
        sim = _sim(_config(self.root, "serverless", 1))
        scaler = sim.getCluster().getScaler()
        inst = scaler.coldStartInstance()
        start = inst.setuptimer
        scaler.advance_indexed_busy_work(sim.getStep(), 1234)
        self.assertEqual(inst.setuptimer, start - 1234)
        cand = scaler.collect_next_event_candidates(sim.getStep())
        self.assertEqual(min(cand), sim.getStep() + max(1, inst.setuptimer - 1))


if __name__ == "__main__":
    unittest.main()
