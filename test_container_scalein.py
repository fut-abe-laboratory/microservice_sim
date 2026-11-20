"""Container scale-in must not leave idle ACTIVE instances outside the balancer."""

import os
import shutil
import tempfile
import unittest

from config import Config
from queue_simulator import QueueSimulator
from status import Status


def _container_config(root, *, num_instances=20, start_instances=4):
    row = [
        "container_scalein_test", "1000", "100000", "req", "102400", "0.2", "0.1", "input",
        "0.1", "15", "0.6", "600", "container", "TRUE", str(num_instances), str(start_instances),
        "30", "30", "100", "1", "-1", "inactive", "verbose", "./result/unused.csv",
    ]
    config = Config()
    config.initialSetup(row, sim_index=0)
    config.OUTPUT_FILE = os.path.join(root, "run_0_result.csv")
    config.OUTPUT_FILE_PACKET = os.path.join(root, "run_0_packet.csv")
    config.OUTPUT_FILE_NUM_INSTANCE = os.path.join(root, "run_0_num_instance.csv")
    config.CONFIG_REQUEST_FILE = os.path.join(root, "requests.csv")
    config.STEADY_ENABLED = False
    config.STEADY_CHECKPOINT_ENABLED = False
    config.SIM_ENABLE_HYBRID_SKIP = True
    config.SIM_ENABLE_SCALE_CHECK_SKIP = True
    return config


def _new_sim(config):
    return QueueSimulator(
        config.SIM_THRESHOLD,
        config.SIM_LIMIT,
        config.SIM_STEP_PER_TIME,
        config.CONFIG_LAMBDA,
        config.CONFIG_MU,
        config,
    )


def _ghosts(sim):
    """ACTIVE/WORKING instances with an armed shutdown timer that the balancer no longer sees."""
    scaler = sim.getCluster().getScaler()
    in_balancer = set(map(id, sim.getCluster().getBalancer().getInstances()))
    return [
        inst
        for inst in scaler.getRunnableInstances()
        if inst.getStatus() == Status.ACTIVE
        and inst.deactivatetimer >= 0
        and id(inst) not in in_balancer
    ]


class TestContainerScaleIn(unittest.TestCase):
    def test_idle_scale_in_reaches_shutdown(self):
        root = tempfile.mkdtemp()
        try:
            sim = _new_sim(_container_config(root))
            scaler = sim.getCluster().getScaler()
            self.assertEqual(sim.getNumOfHotInstance(), 4)

            # All starters are idle, so the first sample scales in to one instance.
            scaler.scaleIn(scaler.getMetrics())
            # Idle ACTIVE victims must be queued for a forced step ...
            self.assertEqual(len(scaler.force_step_instances), 3)
            self.assertTrue(scaler.has_transition_or_busy_immediate(sim.getStep()))

            # ... and one simulated step must move them to SHUTDOWN.
            for instance in tuple(scaler.instances_needing_step(sim.getStep())):
                instance.runStep(sim.getStep())
            self.assertEqual(_ghosts(sim), [])
            statuses = sorted(str(i.getStatus()) for i in scaler.getRunnableInstances())
            self.assertEqual(statuses.count(str(Status.SHUTDOWN)), 3)
            self.assertEqual(sim.getNumOfHotInstance(), 1)
        finally:
            shutil.rmtree(root)

    def test_rebuild_heals_existing_ghosts(self):
        root = tempfile.mkdtemp()
        try:
            sim = _new_sim(_container_config(root))
            scaler = sim.getCluster().getScaler()
            # Recreate the pre-fix state found in checkpoints: deactivated and
            # unregistered, but still ACTIVE and not scheduled for a step.
            victim = next(
                i for i in scaler.getRunnableInstances() if i.getStatus() == Status.ACTIVE
            )
            victim.deactivateInstance()
            scaler.removeInstanceFromBalancer(victim)
            scaler.force_step_instances.clear()
            self.assertEqual(_ghosts(sim), [victim])

            scaler.rebuild_schedule_indexes()
            self.assertIn(victim, scaler.force_step_instances)
            for instance in tuple(scaler.instances_needing_step(sim.getStep())):
                instance.runStep(sim.getStep())
            self.assertEqual(_ghosts(sim), [])
            self.assertEqual(victim.getStatus(), Status.SHUTDOWN)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
