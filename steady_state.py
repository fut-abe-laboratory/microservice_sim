"""Online batch statistics and steady-state detection for QueueSimulator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _z_from_ci_level(ci_level: float) -> float:
    # Common levels; default 95%.
    table = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    return table.get(float(ci_level), 1.959963984540054)


@dataclass
class RunningMoments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def merge_from(self, other: "RunningMoments") -> None:
        if other.n == 0:
            return
        if self.n == 0:
            self.n = other.n
            self.mean = other.mean
            self.m2 = other.m2
            return
        n = self.n + other.n
        delta = other.mean - self.mean
        self.mean = (self.n * self.mean + other.n * other.mean) / n
        self.m2 = self.m2 + other.m2 + delta * delta * self.n * other.n / n
        self.n = n

    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)

    def stddev(self) -> float:
        return math.sqrt(self.variance())

    def ci(self, z: float) -> tuple[float, float]:
        if self.n <= 0:
            return (float("nan"), float("nan"))
        if self.n == 1:
            return (self.mean, self.mean)
        half = z * self.stddev() / math.sqrt(self.n)
        return (self.mean - half, self.mean + half)


@dataclass
class BatchAccumulator:
    t0: float
    lifetimes: RunningMoments = field(default_factory=RunningMoments)
    waits: RunningMoments = field(default_factory=RunningMoments)
    services: RunningMoments = field(default_factory=RunningMoments)
    instance_time_integral: float = 0.0
    last_integral_time: float = 0.0
    last_hot_instances: int = 0

    def __post_init__(self):
        self.last_integral_time = self.t0

    def note_instances(self, sim_time: float, hot_instances: int) -> None:
        if sim_time < self.last_integral_time:
            return
        dt = sim_time - self.last_integral_time
        if dt > 0:
            self.instance_time_integral += self.last_hot_instances * dt
        self.last_integral_time = sim_time
        self.last_hot_instances = hot_instances

    def add_completion(self, lifetime: float, wait: float, service: float) -> None:
        self.lifetimes.add(lifetime)
        self.waits.add(wait)
        self.services.add(service)

    def finalize(self, sim_time: float, z: float) -> dict:
        self.note_instances(sim_time, self.last_hot_instances)
        elapsed = max(sim_time - self.t0, 0.0)
        ave_instances = (
            self.instance_time_integral / elapsed if elapsed > 0 else float(self.last_hot_instances)
        )
        life_ci = self.lifetimes.ci(z)
        wait_ci = self.waits.ci(z)
        svc_ci = self.services.ci(z)
        return {
            "t0": self.t0,
            "t1": sim_time,
            "elapsed": elapsed,
            "n": self.lifetimes.n,
            "ex_time_total": self.lifetimes.mean,
            "ex_time_wait": self.waits.mean,
            "ex_time_service": self.services.mean,
            "ci_total_low": life_ci[0],
            "ci_total_high": life_ci[1],
            "ci_wait_low": wait_ci[0],
            "ci_wait_high": wait_ci[1],
            "ci_service_low": svc_ci[0],
            "ci_service_high": svc_ci[1],
            "ave_instances": ave_instances,
            "_life": self.lifetimes,
            "_wait": self.waits,
            "_service": self.services,
            "_instance_integral": self.instance_time_integral,
        }


def _rel_delta_ok(a: float, b: float, rel_tol: float, eps: float = 1e-12) -> bool:
    return abs(a - b) / max(abs(b), eps) <= rel_tol


def _ci_overlap(lo_a, hi_a, lo_b, hi_b) -> bool:
    if any(map(lambda x: x != x, (lo_a, hi_a, lo_b, hi_b))):  # NaN check
        return False
    return not (hi_a < lo_b or hi_b < lo_a)


def _mean_stable(
    prev: dict,
    curr: dict,
    mean_key: str,
    lo_key: str,
    hi_key: str,
    rel_tol: float,
    eps: float = 1e-12,
) -> bool:
    a = float(prev[mean_key])
    b = float(curr[mean_key])
    return _rel_delta_ok(a, b, rel_tol, eps) or _ci_overlap(
        prev[lo_key], prev[hi_key], curr[lo_key], curr[hi_key]
    )


def _wait_stable(
    prev: dict,
    curr: dict,
    rel_tol: float,
    eps: float = 1e-12,
    wait_abs_tol: float | None = None,
    wait_negligible_frac: float | None = None,
) -> bool:
    """Stability for wait time, with a negligible-wait shortcut.

    When both batch means are tiny relative to sojourn time, relative error on wait
    is dominated by noise (e.g. 1e-7 vs 1e-6 seconds). In that regime we treat wait
    as stable if max(wait) or |Δwait| is below ``frac * min(total)``.
    """
    wa = float(prev["ex_time_wait"])
    wb = float(curr["ex_time_wait"])
    ta = float(prev["ex_time_total"])
    tb = float(curr["ex_time_total"])
    ref_total = max(min(abs(ta), abs(tb)), eps)
    frac = rel_tol if wait_negligible_frac is None else float(wait_negligible_frac)

    if max(wa, wb) <= frac * ref_total:
        return True
    if abs(wa - wb) <= frac * ref_total:
        return True
    if wait_abs_tol is not None and abs(wa - wb) <= float(wait_abs_tol):
        return True

    return _mean_stable(prev, curr, "ex_time_wait", "ci_wait_low", "ci_wait_high", rel_tol, eps)


def metrics_stable(
    prev: dict,
    curr: dict,
    rel_tol: float,
    eps: float = 1e-12,
    wait_abs_tol: float | None = None,
    wait_negligible_frac: float | None = None,
) -> bool:
    """True if batch means are stable for total time, wait, and instance count."""
    if not _mean_stable(
        prev, curr, "ex_time_total", "ci_total_low", "ci_total_high", rel_tol, eps
    ):
        return False
    if not _wait_stable(prev, curr, rel_tol, eps, wait_abs_tol, wait_negligible_frac):
        return False

    ia = float(prev["ave_instances"])
    ib = float(curr["ave_instances"])
    if not _rel_delta_ok(ia, ib, rel_tol, eps):
        return False
    return True


def consecutive_stable(
    batches: list[dict],
    k: int,
    rel_tol: float,
    eps: float = 1e-12,
    wait_abs_tol: float | None = None,
    wait_negligible_frac: float | None = None,
) -> bool:
    if k < 2 or len(batches) < k:
        return False
    window = batches[-k:]
    for i in range(1, len(window)):
        if not metrics_stable(
            window[i - 1],
            window[i],
            rel_tol,
            eps,
            wait_abs_tol=wait_abs_tol,
            wait_negligible_frac=wait_negligible_frac,
        ):
            return False
    return True


def merge_batches(batches: list[dict], z: float) -> dict:
    """Combine K batches into one adopted steady-state summary."""
    if not batches:
        raise ValueError("merge_batches requires at least one batch")

    life = RunningMoments()
    wait = RunningMoments()
    service = RunningMoments()
    integral = 0.0
    t0 = batches[0]["t0"]
    t1 = batches[-1]["t1"]
    for b in batches:
        life.merge_from(b["_life"])
        wait.merge_from(b["_wait"])
        service.merge_from(b["_service"])
        integral += b["_instance_integral"]

    elapsed = max(t1 - t0, 0.0)
    ave_instances = integral / elapsed if elapsed > 0 else batches[-1]["ave_instances"]
    life_ci = life.ci(z)
    wait_ci = wait.ci(z)
    svc_ci = service.ci(z)
    return {
        "t0": t0,
        "t1": t1,
        "elapsed": elapsed,
        "n": life.n,
        "ex_time_total": life.mean,
        "ex_time_wait": wait.mean,
        "ex_time_service": service.mean,
        "ci_total_low": life_ci[0],
        "ci_total_high": life_ci[1],
        "ci_wait_low": wait_ci[0],
        "ci_wait_high": wait_ci[1],
        "ci_service_low": svc_ci[0],
        "ci_service_high": svc_ci[1],
        "ave_instances": ave_instances,
        "num_merged_batches": len(batches),
    }


def resolve_batch_min_time(config) -> float:
    explicit = getattr(config, "STEADY_BATCH_MIN_TIME", None)
    if explicit is not None and float(explicit) > 0:
        return float(explicit)
    from sim_flg import Flg

    if Flg.is_serverless(config.CONFIG_INSTANCE_FLG):
        return float(config.CONFIG_SERVERLESS_TIMER)
    # Container: a few scale intervals as a default dwell window.
    return float(max(config.CONFIG_SCALE_INTERVAL * 4, 1))
