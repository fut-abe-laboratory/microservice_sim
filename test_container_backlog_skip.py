"""A container balancer backlog must not disable hybrid skipping while no
instance can accept work (all ACTIVE/WORKING instances have a queued request)."""

import shutil
import tempfile
import unittest

from request import Request
from test_container_scalein import _container_config, _new_sim


class TestContainerBacklogSkip(unittest.TestCase):
    def test_backlog_is_immediate_only_when_assignable(self):
        root = tempfile.mkdtemp()
        try:
            sim = _new_sim(_container_config(root, start_instances=2))
            balancer = sim.getCluster().getBalancer()
            targets = list(balancer.getInstances())
            self.assertEqual(len(targets), 2)

            balancer.addRequest(Request(9001, 100.0, 0.0))
            # Both instances have an empty queue: the backlog can move now.
            self.assertTrue(balancer.hasAssignableOrScalableWork())

            # Fill every target queue: nothing can be dispatched until a
            # completion, setup end or scale check frees a slot.
            for n, inst in enumerate(targets):
                inst.addRequest(Request(9100 + n, 100.0, 0.0))
            self.assertIsNone(balancer.chooseInstance())
            self.assertFalse(balancer.hasAssignableOrScalableWork())

            # Freeing one queue makes the backlog immediate again.
            targets[0].queue.clear()
            self.assertTrue(balancer.hasAssignableOrScalableWork())
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
