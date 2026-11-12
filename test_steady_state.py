"""Tests for steady-state batch stability detection."""

import unittest

from steady_state import consecutive_stable, metrics_stable


def _batch(total, wait, instances=1.0, n=240000):
    half = 1e-6
    return {
        "ex_time_total": total,
        "ex_time_wait": wait,
        "ave_instances": instances,
        "n": n,
        "ci_total_low": total - half,
        "ci_total_high": total + half,
        "ci_wait_low": max(wait - half, 0.0),
        "ci_wait_high": wait + half,
    }


class TestMetricsStable(unittest.TestCase):
    rel_tol = 0.10

    def test_negligible_wait_fluctuation_is_stable(self):
        """λ=400-like: wait noise should not block convergence."""
        prev = _batch(0.0100254, 4.07859e-08)
        curr = _batch(0.00998619, 3.42617e-07)
        self.assertTrue(metrics_stable(prev, curr, self.rel_tol))

    def test_large_transient_wait_is_unstable(self):
        """λ=800-like: cold-start batch vs steady batch stays unstable."""
        prev = _batch(41.7304, 41.7204, 1.025)
        curr = _batch(0.0100431, 1.18886e-05)
        self.assertFalse(metrics_stable(prev, curr, self.rel_tol))

    def test_steady_batches_after_transient_are_stable(self):
        prev = _batch(0.0100431, 1.18886e-05)
        curr = _batch(0.0100237, 4.94335e-07)
        self.assertTrue(metrics_stable(prev, curr, self.rel_tol))

    def test_consecutive_stable_lambda400_batches(self):
        batches = [
            _batch(0.0100254, 4.07859e-08),
            _batch(0.00998619, 3.42617e-07),
        ]
        self.assertTrue(consecutive_stable(batches, 2, self.rel_tol))

    def test_total_time_drift_still_unstable(self):
        prev = _batch(0.02, 0.001)
        curr = _batch(0.01, 0.001)
        self.assertFalse(metrics_stable(prev, curr, self.rel_tol))


if __name__ == "__main__":
    unittest.main()
