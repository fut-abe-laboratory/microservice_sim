"""Event-driven scaler indexes skip warm-idle instance scans."""

import os
import shutil
import tempfile
import unittest

from config import Config
from queue_simulator import QueueSimulator
from status import Status


def _serverless_config(root, *, num_instances=200, start_instances=50):
    row = [
        "schedule_index_test", "1000", "100000", "req", "102400", "100", "100", "input",
        "1.0", "10", "0.6", "600", "serverless", "TRUE", str(num_instances), "1", "30", "30", "100",
        "1", "-1", "inactive", "verbose", "./result/unused.csv",
    ]
    config = Config()
    config.initialSetup(row, sim_index=0)
    config.CONFIG_DEFAULT_NUM = num_instances
    config.CONFIG_DEFAULT_START_INSTANCES = start_instances
    config.OUTPUT_FILE = os.path.join(root, "run_0_result.csv")
    config.OUTPUT_FILE_PACKET = os.path.join(root, "run_0_packet.csv")
    config.OUTPUT_FILE_NUM_INSTANCE = os.path.join(root, "run_0_num_instance.csv")
    config.CONFIG_REQUEST_FILE = os.path.join(root, "requests.csv")
    config.STEADY_ENABLED = False
    config.STEADY_CHECKPOINT_ENABLED = False
    config.SIM_ENABLE_HYBRID_SKIP = True
    config.SIM_ENABLE_SCALE_CHECK_SKIP = True
    return config


class TestScheduleIndexes(unittest.TestCase):
    def test_hot_count_and_idle_skip(self):
        root = tempfile.mkdtemp()
        try:
            config = _serverless_config(root)
            sim = QueueSimulator(
                config.SIM_THRESHOLD,
                config.SIM_LIMIT,
                config.SIM_STEP_PER_TIME,
                config.CONFIG_LAMBDA,
                config.CONFIG_MU,
                config,
            )
            scaler = sim.getCluster().getScaler()
            self.assertEqual(scaler.hot_count, 50)
            self.assertEqual(sim.getNumOfHotInstance(), 50)

            # After init, warm ACTIVE starters are idle; only force/setup/busy need steps.
            needing = scaler.instances_needing_step(sim.getStep())
            for instance in needing:
                self.assertNotEqual(
                    instance.getStatus(),
                    Status.ACTIVE,
                    msg="warm-idle ACTIVE should not need a step unless forced",
                )

            # Idle heap should schedule deactivation without scanning all runnables.
            next_idle = scaler.next_serverless_idle_step(sim.getStep())
            self.assertIsNotNone(next_idle)
            self.assertGreaterEqual(next_idle, sim.getStep())

            # Hybrid event scan should not depend on walking every runnable.
            self.assertFalse(scaler.has_transition_or_busy_immediate(sim.getStep()))
            candidates = scaler.collect_next_event_candidates(sim.getStep())
            self.assertTrue(candidates)
            self.assertEqual(min(candidates), next_idle)
        finally:
            shutil.rmtree(root)

    def test_busy_instance_is_stepped(self):
        root = tempfile.mkdtemp()
        try:
            config = _serverless_config(root, num_instances=20, start_instances=5)
            sim = QueueSimulator(
                config.SIM_THRESHOLD,
                config.SIM_LIMIT,
                config.SIM_STEP_PER_TIME,
                config.CONFIG_LAMBDA,
                config.CONFIG_MU,
                config,
            )
            scaler = sim.getCluster().getScaler()
            active = next(
                inst for inst in scaler.getInstances() if inst.getStatus() == Status.ACTIVE
            )
            from request import Request

            req = Request(0, 1.0 / config.CONFIG_MU, 0.0)
            active.addRequest(req)
            self.assertIn(active, scaler.busy_instances)
            needing = scaler.instances_needing_step(sim.getStep())
            self.assertIn(active, needing)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
