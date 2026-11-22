"""Wall-clock checkpoints so a killed steady-state run can resume.

Packet and instance logs are not a simulator state. This module stores the
mutable event-loop state (step, in-flight requests, instance timers, batch
statistics, RNG) and restores it onto a freshly loaded arrival list.

Version 2 stores only materialized (non-INACTIVE) instances for a lazy pool.
Older checkpoint files are unsupported; delete them and restart incomplete runs.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from status import Status

CHECKPOINT_VERSION = 2


def checkpoint_path_for(result_csv_path: str) -> str:
    if result_csv_path.endswith("_result.csv"):
        return result_csv_path[: -len("_result.csv")] + "_checkpoint.pkl"
    return result_csv_path + ".checkpoint.pkl"


def repair_text_tail(path: str) -> None:
    """Drop a torn final CSV row left by a hard kill."""
    if not os.path.isfile(path):
        return
    size = os.path.getsize(path)
    if size <= 0:
        return
    with open(path, "rb+") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
        chunk = min(size, 1024 * 1024)
        handle.seek(size - chunk)
        data = handle.read(chunk)
        cut = data.rfind(b"\n")
        if cut < 0:
            handle.truncate(0)
            return
        handle.truncate(size - chunk + cut + 1)


def _snap_request(req) -> dict:
    return {
        "id": int(req.getId()),
        "org_workload": float(req.getOrgWorkload()),
        "workload": float(req.getWorkload()),
        "time_start": float(req.getStartTime()),
        "time_start_process": float(req.getStartProcessTime()),
        "time_end": float(req.getEndTime()),
        "status": int(req.getStatus()),
        "processed_by": req.getProcessedBy(),
    }


def _apply_request(req, snap: dict) -> None:
    req.org_workload = float(snap["org_workload"])
    req.setWorkload(float(snap["workload"]))
    req.setStartTime(float(snap["time_start"]))
    req.setStartProcessTime(float(snap["time_start_process"]))
    req.setEndTime(float(snap["time_end"]))
    req.setStatus(int(snap["status"]))
    req.setProcessedBy(snap["processed_by"])


def _snap_instance(instance) -> dict:
    from collections import deque

    queue = instance.queue
    if isinstance(queue, deque):
        queued = list(queue)
    else:
        queued = list(queue)
    return {
        "id": int(instance.getId()),
        "status": int(instance.getStatus()),
        "setuptimer": instance.setuptimer,
        "deactivatetimer": instance.deactivatetimer,
        "cpu_ctr": instance.getCpuCtr(),
        "last_time": getattr(instance, "last_time", None),
        "scaler_attached": getattr(instance, "scaler", None) is not None,
        "queue": [_snap_request(req) for req in queued],
        "exec_queue": [
            None if req is None else _snap_request(req) for req in instance.exec_queue
        ],
    }


def capture_state(sim, before_instances) -> dict:
    cluster = sim.getCluster()
    scaler = cluster.getScaler()
    balancer = cluster.getBalancer()
    sender = sim.getSender()
    generator = sim.getGenerator()
    next_unsent_id = None
    if sender.request_ptr < len(sender.reqs):
        next_unsent_id = int(sender.reqs[sender.request_ptr].getId())

    # Sparse: only materialized non-INACTIVE instances.
    instance_snaps = []
    for instance in scaler.getInstances():
        if instance.getStatus() == Status.INACTIVE:
            continue
        instance_snaps.append(_snap_instance(instance))

    return {
        "version": CHECKPOINT_VERSION,
        "timestep": int(sim.getStep()),
        "num_reqs": int(sim.num_reqs),
        "next_request_time": sim.next_request_time,
        "before_instances": before_instances,
        "request_ptr_next_id": next_unsent_id,
        "packet_streamed": bool(getattr(sim, "_packet_streamed", False)),
        "steady_batches": list(sim.steady_batches),
        "steady_acc": sim.steady_acc,
        "steady_result": sim.steady_result,
        "steady_reached": bool(sim.steady_reached),
        "generator": {
            "next_arrival_time": generator.next_arrival_time,
            "next_id": generator.next_id,
            "max_loaded_id": generator.max_loaded_id,
            "last_start_time": generator.last_start_time,
        },
        "rng_state": np.random.get_state(),
        "balancer": {
            "last_instance": balancer.last_instance,
            "instance_ids": [int(inst.getId()) for inst in balancer.getInstances()],
            "requests": [_snap_request(req) for req in balancer.getRequests()],
        },
        "scaler": {
            "num_active_instance": scaler.num_active_instance,
            "runnable_ids": [int(inst.getId()) for inst in scaler.getRunnableInstances()],
            "pool_capacity": int(getattr(scaler, "pool_capacity", 0) or 0),
        },
        "cluster_num_active_cpu": cluster.num_active_CPU,
        "cluster_pool_capacity": int(getattr(cluster, "pool_capacity", 0) or 0),
        "instances": instance_snaps,
    }


def _index_requests(sim) -> dict:
    found = {}
    for req in sim.getGenerator().reqs:
        found[int(req.getId())] = req
    return found


def _resolve_request(found: dict, snap: dict):
    req_id = int(snap["id"])
    req = found.get(req_id)
    if req is None:
        raise KeyError(f"checkpoint request id {req_id} is not in the reloaded arrival list")
    _apply_request(req, snap)
    return req


def apply_state(sim, state: dict) -> int:
    if int(state.get("version", 0)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version: {state.get('version')} "
            f"(need {CHECKPOINT_VERSION}; delete old *_checkpoint.pkl and restart)"
        )

    found = _index_requests(sim)
    cluster = sim.getCluster()
    scaler = cluster.getScaler()
    balancer = cluster.getBalancer()
    sender = sim.getSender()
    generator = sim.getGenerator()

    sim.setStep(int(state["timestep"]))
    sim.num_reqs = int(state["num_reqs"])
    sim.next_request_time = state["next_request_time"]
    sim._packet_streamed = bool(state["packet_streamed"])
    sim.steady_batches = list(state["steady_batches"])
    sim.steady_acc = state["steady_acc"]
    sim.steady_result = state["steady_result"]
    sim.steady_reached = bool(state["steady_reached"])

    gen = state["generator"]
    if int(gen["max_loaded_id"]) >= int(generator.max_loaded_id):
        generator.next_arrival_time = gen["next_arrival_time"]
        generator.next_id = gen["next_id"]
        generator.max_loaded_id = gen["max_loaded_id"]
        generator.last_start_time = gen["last_start_time"]
    np.random.set_state(state["rng_state"])

    next_id = state["request_ptr_next_id"]
    if next_id is None:
        sender.request_ptr = len(sender.reqs)
    else:
        sender.request_ptr = None
        for index, req in enumerate(sender.reqs):
            if int(req.getId()) == int(next_id):
                sender.request_ptr = index
                break
        if sender.request_ptr is None:
            raise KeyError(f"checkpoint next request id {next_id} is missing")
    if sender.request_ptr < len(sender.reqs):
        sender.setNextRequestTime(sender.reqs[sender.request_ptr].getStartTime())

    sc = state["scaler"]
    pool_capacity = int(sc.get("pool_capacity") or state.get("cluster_pool_capacity") or 0)
    if pool_capacity <= 0:
        pool_capacity = int(getattr(scaler, "pool_capacity", 0) or 0)
    if pool_capacity <= 0:
        raise ValueError("checkpoint missing pool_capacity")

    # Reset lazy pool, then materialize snapped instances.
    if scaler._instance_factory is None:
        raise RuntimeError("instance factory missing on resume")
    scaler.init_pool(pool_capacity)
    scaler._cluster = cluster
    cluster.pool_capacity = int(state.get("cluster_pool_capacity") or pool_capacity)
    cluster.instances = []
    cluster.num_active_CPU = state["cluster_num_active_cpu"]
    balancer.instances = []
    balancer.requests = []

    from collections import deque

    for inst_id in range(pool_capacity):
        scaler._push_free_id(inst_id)

    by_id = {}
    for snap in state["instances"]:
        status = int(snap["status"])
        if status == Status.INACTIVE:
            continue
        inst = scaler.materialize_instance(int(snap["id"]), status, from_free=True)
        # Direct field restore (avoid duplicate setStatus side effects).
        inst.status = status
        inst.setuptimer = snap["setuptimer"]
        inst.deactivatetimer = snap["deactivatetimer"]
        inst.setCpuCtr(snap["cpu_ctr"])
        if snap["last_time"] is not None and hasattr(inst, "last_time"):
            inst.last_time = snap["last_time"]
        inst.scaler = scaler if snap["scaler_attached"] else None
        inst.queue = deque(_resolve_request(found, req) for req in snap["queue"])
        restored_exec = []
        for req_snap in snap["exec_queue"]:
            restored_exec.append(None if req_snap is None else _resolve_request(found, req_snap))
        if len(restored_exec) != len(inst.exec_queue):
            raise ValueError(
                f"exec_queue length mismatch for instance {snap['id']}: "
                f"{len(restored_exec)} vs {len(inst.exec_queue)}"
            )
        inst.exec_queue = restored_exec
        by_id[int(inst.getId())] = inst

    bal = state["balancer"]
    balancer.last_instance = bal["last_instance"]
    balancer.instances = [by_id[int(i)] for i in bal["instance_ids"]]
    balancer.requests = [_resolve_request(found, snap) for snap in bal["requests"]]
    if hasattr(balancer, "rebuild_hot_index"):
        balancer.rebuild_hot_index()

    scaler.num_active_instance = sc["num_active_instance"]
    scaler.runnable_instances = [by_id[int(i)] for i in sc["runnable_ids"]]
    scaler.runnable_instance_set = set(scaler.runnable_instances)

    if hasattr(scaler, "rebuild_schedule_indexes"):
        scaler.rebuild_schedule_indexes()

    sim._resumed = True
    return state["before_instances"]


def save_checkpoint(sim, before_instances) -> str:
    path = sim.checkpoint_path
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    payload = capture_state(sim, before_instances)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def load_checkpoint(sim) -> int | None:
    path = sim.checkpoint_path
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return None
    with open(path, "rb") as handle:
        state = pickle.load(handle)
    return apply_state(sim, state)


def discard_checkpoint(sim) -> None:
    path = getattr(sim, "checkpoint_path", None)
    if path and os.path.isfile(path):
        os.remove(path)
