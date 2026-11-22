"""hottest() index, staggered serverless starters and the full-pool balancer
short-cut. The indexed hottest() must always return the same instance as the
linear scan (max last_time, ties -> earliest registered)."""

import os
import shutil
import tempfile
import unittest

from config import Config
from queue_simulator import QueueSimulator
from request import Request
from status import Status
import balancer as balancer_mod


def _config(root, mode, *, pool=1000, start=0, setup_sec=1, lam=50, mu=10, step_per_time=100000):
    row = [
        "hot_index_test", "1000", str(step_per_time), "req", str(pool), str(lam), str(mu), "input",
        "0.1", "15", "0.6", "600", mode, "TRUE", str(pool), str(start),
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


class TestServerlessHotIndex(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_staggered_starters_are_reachable(self):
        for mode in ("serverless", "serverless_warm_wait"):
            sim = _sim(_config(self.root, mode, start=5))
            bal = sim.getCluster().getBalancer()
            starters = sorted(bal.getInstances(), key=lambda i: i.getId())
            self.assertEqual(len(starters), 5)
            self.assertTrue(all(i.getLastTime() < 0 for i in starters[1:]))
            # Most recently used first, then the next-youngest idle age.
            for k, inst in enumerate(starters):
                self.assertIs(bal.hottest(), inst, (mode, k))
                self.assertIs(bal._hottest_scan(), inst)
                inst.addRequest(Request(k, 1.0, 0.0))
            self.assertIsNone(bal.hottest())
            self.assertIsNone(bal._hottest_scan())

    def test_index_matches_scan_during_run(self):
        calls = [0]
        orig = balancer_mod.Balancer.hottest

        def checked(self):
            got = orig(self)
            if self._uses_hot_index():
                calls[0] += 1
                assert got is self._hottest_scan()
            return got

        balancer_mod.Balancer.hottest = checked
        try:
            for mode in ("serverless", "serverless_warm_wait"):
                sim = _sim(_config(self.root, mode, pool=12, start=6, setup_sec=2,
                                   lam=40, mu=10, step_per_time=1000))
                sim.startSimulate()
                self.assertGreater(sim.countReqs(), 500)
        finally:
            balancer_mod.Balancer.hottest = orig
        self.assertGreater(calls[0], 1000)

    def test_full_pool_stops_after_first_failed_request(self):
        for mode in ("serverless", "serverless_warm_wait"):
            sim = _sim(_config(self.root, mode, pool=3, setup_sec=5))
            cluster = sim.getCluster()
            bal, scaler = cluster.getBalancer(), cluster.getScaler()
            for k in range(3):
                bal.addRequest(Request(k, 1.0, 0.0))
            bal.manageQueue()  # three cold starts fill the pool
            self.assertFalse(scaler.hasFreeSlot())
            for k in range(3, 103):
                bal.addRequest(Request(k, 1.0, 0.0))
            tries = [0]
            orig = bal.chooseInstance

            def counting(queue_len=None):
                tries[0] += 1
                return orig(queue_len=queue_len)

            bal.chooseInstance = counting
            bal.manageQueue()
            self.assertEqual(tries[0], 1, mode)
            self.assertEqual(len(bal.getRequests()), 100)
            self.assertFalse(bal.hasAssignableOrScalableWork())

    def _warm_wait_queue(self, setup_sec):
        sim = _sim(_config(self.root, "serverless_warm_wait", start=2, setup_sec=setup_sec, mu=10))
        cluster = sim.getCluster()
        bal = cluster.getBalancer()
        for k, inst in enumerate(list(bal.getInstances())):
            inst.addRequest(Request(1000 + k, 1.0, 0.0))  # both starters busy
        for k in range(50):
            bal.addRequest(Request(k, 1.0, 0.0))
        tries = [0]
        orig = bal.chooseInstance

        def counting(queue_len=None):
            tries[0] += 1
            return orig(queue_len=queue_len)

        bal.chooseInstance = counting
        bal.manageQueue()
        return bal, tries[0]

    def test_warm_wait_skips_requests_that_still_wait(self):
        # absorb = S * W * mu = 100 * 2 * 10 = 2000 >= 50: every request waits.
        bal, tries = self._warm_wait_queue(100)
        self.assertEqual(tries, 1)
        self.assertEqual([r.getId() for r in bal.getRequests()], list(range(50)))
        self.assertEqual(len(bal.getInstances()), 2)

    def test_warm_wait_skip_then_cold_start_like_linear_loop(self):
        # absorb = 1 * 2 * 10 = 20: positions 1..20 wait, the other 30 cold-start.
        bal, tries = self._warm_wait_queue(1)
        self.assertEqual(tries, 1 + 30)
        self.assertEqual([r.getId() for r in bal.getRequests()], list(range(20)))
        self.assertEqual(len(bal.getInstances()), 2 + 30)

    def _run_trace(self, mode, skip, request_file=None, **kw):
        root = tempfile.mkdtemp(dir=self.root)
        config = _config(root, mode, **kw)
        config.SIM_ENABLE_HYBRID_SKIP = skip
        if request_file is not None:
            shutil.copy(request_file, config.CONFIG_REQUEST_FILE)
        sim = _sim(config)
        sim.startSimulate()
        trace = [(r.getId(), r.processedBy, r.time_end) for r in sim.reqs]
        return trace, config.CONFIG_REQUEST_FILE

    def test_event_driven_matches_per_step(self):
        # A just-finished serverless instance stays WORKING until its next
        # executed step; skipping past that step used to hide it from
        # hottest() and cause extra cold starts.
        for mode in ("serverless", "serverless_warm_wait"):
            for start in (0, 3):
                kw = dict(pool=50, start=start, setup_sec=2, lam=3, mu=1, step_per_time=1000)
                want, req_file = self._run_trace(mode, False, **kw)
                got, _ = self._run_trace(mode, True, request_file=req_file, **kw)
                self.assertGreater(len(want), 500)
                self.assertEqual(got, want, (mode, start))

if __name__ == "__main__":
    unittest.main()
