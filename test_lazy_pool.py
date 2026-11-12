"""Lazy instance pool + free-list preserve activation order."""

import os
import shutil
import tempfile
import unittest

from config import Config
from queue_simulator import QueueSimulator
from status import Status
from checkpoint import CHECKPOINT_VERSION, capture_state, save_checkpoint, load_checkpoint


def _serverless_config(root, *, num_instances=20, start_instances=2):
    row = [
        "lazy_pool_test", "1000", "1000", "req", "102400", "1", "10", "input",
        "0.1", "15", "0.6", "10", "serverless", "TRUE", str(num_instances), "0", "1", "1", "100",
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
    config.STEADY_CHECKPOINT_ENABLED = True
    config.STEADY_CHECKPOINT_INTERVAL_SEC = 10**9
    config.SIM_ENABLE_HYBRID_SKIP = True
    return config


class TestLazyPool(unittest.TestCase):
    def test_only_starters_materialized(self):
        root = tempfile.mkdtemp()
        try:
            config = _serverless_config(root, num_instances=50, start_instances=3)
            sim = QueueSimulator(
                config.SIM_THRESHOLD,
                config.SIM_LIMIT,
                config.SIM_STEP_PER_TIME,
                config.CONFIG_LAMBDA,
                config.CONFIG_MU,
                config,
            )
            scaler = sim.getCluster().getScaler()
            self.assertEqual(scaler.pool_capacity, 50)
            self.assertEqual(len(scaler.getInstances()), 3)
            self.assertEqual(sim.getCluster().getNumOfInstances(), 50)
            self.assertEqual(len(scaler._free_set), 47)
            for i in range(3):
                self.assertIsNotNone(scaler.get_slot(i))
                self.assertEqual(scaler.get_slot(i).getStatus(), Status.ACTIVE)
            self.assertIsNone(scaler.get_slot(3))
        finally:
            shutil.rmtree(root)

    def test_cold_start_uses_lowest_free_id(self):
        root = tempfile.mkdtemp()
        try:
            config = _serverless_config(root, num_instances=10, start_instances=2)
            sim = QueueSimulator(
                config.SIM_THRESHOLD,
                config.SIM_LIMIT,
                config.SIM_STEP_PER_TIME,
                config.CONFIG_LAMBDA,
                config.CONFIG_MU,
                config,
            )
            scaler = sim.getCluster().getScaler()
            inst = scaler.coldStartInstance()
            self.assertIsNotNone(inst)
            self.assertEqual(inst.getId(), 2)
            self.assertEqual(inst.getStatus(), Status.SETUP)
            self.assertEqual(len(scaler.getInstances()), 3)
        finally:
            shutil.rmtree(root)

    def test_checkpoint_is_sparse_v2(self):
        root = tempfile.mkdtemp()
        try:
            config = _serverless_config(root, num_instances=30, start_instances=2)
            sim = QueueSimulator(
                config.SIM_THRESHOLD,
                config.SIM_LIMIT,
                config.SIM_STEP_PER_TIME,
                config.CONFIG_LAMBDA,
                config.CONFIG_MU,
                config,
            )
            state = capture_state(sim, before_instances=2)
            self.assertEqual(state["version"], CHECKPOINT_VERSION)
            self.assertEqual(state["scaler"]["pool_capacity"], 30)
            self.assertEqual(len(state["instances"]), 2)

            save_checkpoint(sim, 2)
            restored = QueueSimulator(
                config.SIM_THRESHOLD,
                config.SIM_LIMIT,
                config.SIM_STEP_PER_TIME,
                config.CONFIG_LAMBDA,
                config.CONFIG_MU,
                config,
            )
            self.assertTrue(restored._resumed)
            rscaler = restored.getCluster().getScaler()
            self.assertEqual(rscaler.pool_capacity, 30)
            self.assertEqual(len(rscaler.getInstances()), 2)
            self.assertEqual(len(rscaler._free_set), 28)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
