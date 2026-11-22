from status import Status
from config import Config
import heapq
import math
from sim_flg import Flg
from serverless import Serverless


class Scaler:

    def __init__(self, balancer, config: Config):
        self.config = config
        self.setBalancer(balancer)
        # Fixed-capacity pool: only non-INACTIVE instances are materialized.
        self.pool_capacity = 0
        self._slots = []  # index = instance id; None => free/INACTIVE
        self._free_set = set()
        self._free_heap = []  # min-heap of free ids (lazy deletion vs _free_set)
        self._instance_factory = None  # fn(status) -> Instance
        self.instances = []  # materialized only (compat / iteration helpers)
        self.runnable_instances = []
        self.runnable_instance_set = set()
        self.num_active_instance = 0
        self.mode = self.config.CONFIG_INSTANCE_FLG
        # Event-driven indexes: avoid scanning every warm-idle instance each step.
        self.setup_instances = set()
        self.shutdown_instances = set()
        self.busy_instances = set()
        self.force_step_instances = set()
        self.hot_count = 0
        self._idle_heap = []  # (deadline_step, last_time, inst_id)
        # Virtual setup/shutdown timers: one global pass counter plus a heap of
        # (due_pass, token, id, oid, instance) instead of stepping every SETUP /
        # SHUTDOWN instance on every executed step.
        self.timer_passes = 0
        self._timer_heap = []
        return

    def set_instance_factory(self, factory):
        """factory(status) -> Instance (id unset)."""
        self._instance_factory = factory

    def init_pool(self, capacity: int):
        capacity = int(capacity)
        if capacity < 0:
            raise ValueError("pool capacity must be non-negative")
        self.pool_capacity = capacity
        self._slots = [None] * capacity
        self._free_set = set()
        self._free_heap = []
        self.instances = []

    def _push_free_id(self, inst_id: int):
        inst_id = int(inst_id)
        if inst_id in self._free_set:
            return
        self._free_set.add(inst_id)
        heapq.heappush(self._free_heap, inst_id)

    def _take_free_id(self, inst_id: int | None = None):
        if inst_id is not None:
            inst_id = int(inst_id)
            if inst_id not in self._free_set:
                return None
            self._free_set.remove(inst_id)
            return inst_id
        while self._free_heap:
            cand = heapq.heappop(self._free_heap)
            if cand in self._free_set:
                self._free_set.remove(cand)
                return cand
        return None

    def materialize_instance(self, inst_id: int, status, *, from_free: bool = False):
        """Create instance at id and register it.

        from_free=True requires the id to be on the free-list (scale-out / cold-start).
        from_free=False is for initial starters (id not yet on the free-list).
        """
        inst_id = int(inst_id)
        if inst_id < 0 or inst_id >= self.pool_capacity:
            raise ValueError(f"instance id {inst_id} out of pool range")
        if self._slots[inst_id] is not None:
            raise ValueError(f"slot {inst_id} is already materialized")
        if from_free:
            if self._take_free_id(inst_id) is None:
                raise ValueError(f"id {inst_id} is not free")
        elif inst_id in self._free_set:
            raise ValueError(f"id {inst_id} unexpectedly marked free")
        if self._instance_factory is None:
            raise RuntimeError("instance factory is not set")
        instance = self._instance_factory(status)
        instance.setId(inst_id)
        self._slots[inst_id] = instance
        self.instances.append(instance)
        instance.setStatusManager(self)
        cluster = getattr(self, "_cluster", None)
        if cluster is not None and hasattr(cluster, "noteMaterializedInstance"):
            cluster.noteMaterializedInstance(instance)
        if instance.getStatus() != Status.INACTIVE:
            self.registerRunnableInstance(instance)
        if instance.getStatus() == Status.ACTIVE:
            self.num_active_instance += 1
            self.registerInstance2Balancer(instance)
        self.refresh_instance_schedule(instance)
        return instance

    def release_inactive_instance(self, instance):
        """Dematerialize an INACTIVE instance and return its id to the free-list."""
        if instance.getStatus() != Status.INACTIVE:
            return
        inst_id = int(instance.getId())
        if inst_id < 0 or inst_id >= self.pool_capacity:
            return
        if self._slots[inst_id] is not instance:
            return
        self.removeRunnableInstance(instance)
        self._slots[inst_id] = None
        if instance in self.instances:
            self.instances.remove(instance)
        instance.setStatusManager(None)
        self._push_free_id(inst_id)
        # Keep cluster.instances in sync when present.
        cluster = getattr(self, "_cluster", None)
        if cluster is not None and hasattr(cluster, "noteDematerializedInstance"):
            cluster.noteDematerializedInstance(instance)

    def get_slot(self, inst_id: int):
        return self._slots[int(inst_id)]

    def runStep(self, timestep):
        if self.mode == Flg.FLG_CONTAINER:
            self.runStep4Container(timestep)
        elif Flg.is_serverless(self.mode):
            self.runStep4Serverless(timestep)

    def runStep4Container(self, timestep):
        if not self.isTime2Check(timestep):
            return
        if not self.containerScaleCheckNeeded():
            return

        metrics = self.getMetrics()
        if metrics - 1 > self.config.CONFIG_SCALE_SENSITIVE:
            self.scaleOut(metrics)
        elif 1 - metrics > self.config.CONFIG_SCALE_SENSITIVE:
            self.scaleIn(metrics)
        return

    def runStep4Serverless(self, timestep):
        # In warm-wait mode, keep warm instances while requests are queued at the balancer.
        defer_idle = (
            self.mode == Flg.FLG_SERVERLESS_WARM_WAIT
            and bool(self.getBalancer().getRequests())
        )
        if defer_idle:
            return

        idle_steps = self._serverless_idle_steps()
        due = []
        seen = set()
        while self._idle_heap:
            deadline, last_time, inst_id = self._idle_heap[0]
            if deadline > timestep:
                break
            heapq.heappop(self._idle_heap)
            instance = self.get_slot(inst_id)
            if instance is None or instance.getStatus() != Status.ACTIVE:
                continue
            if int(getattr(instance, "last_time", -1)) != last_time:
                continue
            if timestep - last_time < idle_steps:
                continue
            # Heap may hold duplicate (deadline, last_time) entries for one instance.
            if inst_id in seen:
                continue
            seen.add(inst_id)
            due.append(instance)
        for instance in due:
            self.deactivateOldInstance(instance)

    def setBalancer(self, balancer):
        self.balancer = balancer
        return

    def getBalancer(self):
        return self.balancer

    def addInstance(self, instance):
        """Legacy append path (non-pool configs). Prefer materialize_instance."""
        self.instances.append(instance)
        if self.pool_capacity <= 0:
            # Keep denselist mode working for CONFIG_INSTANCES rows.
            while len(self._slots) <= int(instance.getId()):
                self._slots.append(None)
            self.pool_capacity = len(self._slots)
        inst_id = int(instance.getId())
        if 0 <= inst_id < len(self._slots):
            self._slots[inst_id] = instance
            self._free_set.discard(inst_id)
        instance.setStatusManager(self)
        if instance.getStatus() != Status.INACTIVE:
            self.registerRunnableInstance(instance)
        if instance.getStatus() == Status.ACTIVE:
            self.num_active_instance += 1
            self.registerInstance2Balancer(instance)
        self.refresh_instance_schedule(instance)
        return

    def getInstances(self):
        """Materialized instances only (INACTIVE are dematerialized)."""
        return self.instances

    def getPoolCapacity(self):
        return int(self.pool_capacity)

    def getRunnableInstances(self):
        return self.runnable_instances

    def registerRunnableInstance(self, instance):
        if instance not in self.runnable_instance_set:
            self.runnable_instances.append(instance)
            self.runnable_instance_set.add(instance)
        return

    def removeRunnableInstance(self, instance):
        if instance in self.runnable_instance_set:
            self.runnable_instance_set.remove(instance)
            self.runnable_instances.remove(instance)
        self.setup_instances.discard(instance)
        self.shutdown_instances.discard(instance)
        self.busy_instances.discard(instance)
        self.force_step_instances.discard(instance)
        if getattr(instance, "_sched_hot", False):
            self.hot_count = max(0, self.hot_count - 1)
            instance._sched_hot = False
        return

    def registerInstance2Balancer(self, instance):
        self.getBalancer().addInstance(instance)
        return

    def removeInstanceFromBalancer(self, instance):
        self.getBalancer().delInstance(instance)
        return

    def _instance_has_work(self, instance):
        if instance.getQueueLength() > 0:
            return True
        for req in instance.exec_queue:
            if req is not None:
                return True
        return False

    def _serverless_idle_steps(self):
        return int(self.config.CONFIG_SERVERLESS_TIMER * self.config.SIM_STEP_PER_TIME)

    def note_serverless_last_time(self, instance, last_time):
        if not Flg.is_serverless(self.mode):
            return
        self.getBalancer().note_hot_candidate(instance)
        if instance.getStatus() != Status.ACTIVE:
            return
        if self._instance_has_work(instance):
            return
        deadline = int(last_time) + self._serverless_idle_steps()
        heapq.heappush(
            self._idle_heap,
            (deadline, int(last_time), int(instance.getId())),
        )

    def refresh_instance_schedule(self, instance):
        """Reclassify instance into setup/shutdown/busy indexes and hot_count."""
        status = instance.getStatus()
        if status == Status.ACTIVE:
            self.getBalancer().note_hot_candidate(instance)

        if status == Status.SETUP:
            self.setup_instances.add(instance)
        else:
            self.setup_instances.discard(instance)

        if status == Status.SHUTDOWN:
            self.shutdown_instances.add(instance)
        else:
            self.shutdown_instances.discard(instance)

        has_work = self._instance_has_work(instance)
        if status in (Status.SETUP, Status.SHUTDOWN):
            # Timer-driven: scheduled through the timer heap, not stepped.
            self.busy_instances.discard(instance)
        elif has_work or status == Status.WORKING:
            self.busy_instances.add(instance)
        else:
            self.busy_instances.discard(instance)

        hot_now = status in (Status.ACTIVE, Status.WORKING)
        prev_hot = getattr(instance, "_sched_hot", False)
        if hot_now and not prev_hot:
            self.hot_count += 1
        elif prev_hot and not hot_now:
            self.hot_count = max(0, self.hot_count - 1)
        instance._sched_hot = hot_now

        if (
            Flg.is_serverless(self.mode)
            and status == Status.ACTIVE
            and not has_work
            and hasattr(instance, "last_time")
            and instance.last_time is not None
        ):
            # last_time may be negative for staggered initial idle ages.
            self.note_serverless_last_time(instance, instance.last_time)

    def on_instance_status_changed(self, instance, prev_status, status):
        if status == Status.INACTIVE:
            self.release_inactive_instance(instance)
            return
        self.refresh_instance_schedule(instance)

    def _timer_due_pass(self, instance):
        status = instance.getStatus()
        anchor = instance._timer_anchor
        if anchor is None:
            return None
        if status == Status.SETUP:
            # Serverless flips at timer <= 0, Container at timer < 0.
            if isinstance(instance, Serverless):
                return anchor + instance._setuptimer
            return anchor + instance._setuptimer + 1
        if status == Status.SHUTDOWN:
            return anchor + instance._deactivatetimer + 1
        return None

    def schedule_timer(self, instance):
        due = self._timer_due_pass(instance)
        if due is not None:
            heapq.heappush(
                self._timer_heap,
                (due, instance._timer_token, int(instance.getId()), id(instance), instance),
            )

    def _timer_entry_valid(self, entry):
        _due, token, _iid, _oid, instance = entry
        if instance._timer_token != token or instance._timer_anchor is None:
            return False
        status = instance.getStatus()
        if status == Status.SETUP:
            return instance in self.setup_instances
        if status == Status.SHUTDOWN:
            return instance in self.shutdown_instances
        return False

    def min_timer_due(self):
        heap = self._timer_heap
        while heap:
            if self._timer_entry_valid(heap[0]):
                return heap[0][0]
            heapq.heappop(heap)
        return None

    def rebuild_schedule_indexes(self):
        self.setup_instances = set()
        self.shutdown_instances = set()
        self._timer_heap = []
        self.busy_instances = set()
        self.force_step_instances = set()
        self.hot_count = 0
        self._idle_heap = []
        for instance in self.getRunnableInstances():
            instance._sched_hot = False
            if instance.getStatus() in (Status.SETUP, Status.SHUTDOWN):
                instance._materialize_timer()
            self.refresh_instance_schedule(instance)
            if instance.getStatus() in (Status.SETUP, Status.SHUTDOWN):
                instance._anchor_timer()
            # Heal checkpoints taken while a scaled-in instance was still
            # ACTIVE/WORKING with an armed deactivate timer.
            if (
                instance.getStatus() in (Status.ACTIVE, Status.WORKING)
                and instance.deactivatetimer >= 0
            ):
                self.force_step_instances.add(instance)

    def instances_needing_step(self, timestep):
        """Instances that must run this step (skip warm-idle ACTIVE)."""
        # One pass = the per-step decrement every SETUP/SHUTDOWN timer used to get.
        self.timer_passes += 1
        due = set(self.busy_instances)
        due.update(self.force_step_instances)
        self.force_step_instances.clear()
        heap = self._timer_heap
        passes = self.timer_passes
        while heap and heap[0][0] <= passes:
            entry = heapq.heappop(heap)
            if self._timer_entry_valid(entry):
                due.add(entry[4])
        # Deterministic, id-ascending order: same-step events (e.g. several
        # setups finishing together) register to the balancer lowest id first,
        # independent of object addresses.
        return sorted(due, key=lambda i: i.getId())

    def next_serverless_idle_step(self, step, defer_idle=False):
        if defer_idle or not Flg.is_serverless(self.mode):
            return None
        while self._idle_heap:
            deadline, last_time, inst_id = self._idle_heap[0]
            instance = self.get_slot(inst_id)
            if instance is None or instance.getStatus() != Status.ACTIVE:
                heapq.heappop(self._idle_heap)
                continue
            if int(getattr(instance, "last_time", -1)) != last_time:
                heapq.heappop(self._idle_heap)
                continue
            if deadline <= step:
                return step
            return deadline
        return None

    def needs_idle_flip(self, instance):
        """Serverless WORKING instance with nothing left to run.

        processRequests() flips it to ACTIVE on its next executed step; only
        then can hottest() pick it. Per-step simulation runs that step right
        away, so event-driven skipping must not jump past it (container
        dispatch accepts WORKING instances, so it is unaffected).
        """
        if not Flg.is_serverless(self.mode):
            return False
        if instance.getStatus() != Status.WORKING or instance.queue:
            return False
        for slot in instance.exec_queue:
            if slot is not None:
                return False
        return True

    def has_transition_or_busy_immediate(self, step):
        if self.force_step_instances:
            return True
        due = self.min_timer_due()
        if due is not None and due <= self.timer_passes + 1:
            return True
        for instance in self.busy_instances:
            status = instance.getStatus()
            if status in (Status.SETUP, Status.SHUTDOWN):
                continue
            exec_queue = instance.exec_queue
            queue = instance.queue
            if queue and any(slot is None for slot in exec_queue):
                return True
            for req in exec_queue:
                if req is not None and req.workload <= 0:
                    return True
            if self.needs_idle_flip(instance):
                return True
        idle_due = self.next_serverless_idle_step(step, defer_idle=False)
        return idle_due is not None and idle_due <= step

    def collect_next_event_candidates(self, step, defer_idle=False):
        candidates = []
        due = self.min_timer_due()
        if due is not None:
            # The step whose pass reaches `due` (passes advance one per step).
            candidates.append(step + max(1, due - self.timer_passes - 1))
        for instance in self.busy_instances:
            status = instance.getStatus()
            if status in (Status.SETUP, Status.SHUTDOWN):
                continue
            if status not in (Status.ACTIVE, Status.WORKING):
                continue
            for req in instance.exec_queue:
                if req is None:
                    continue
                if req.workload <= 0:
                    candidates.append(step)
                    continue
                processing_steps = int(math.ceil(req.workload / instance.processing_capacity))
                candidates.append(step + processing_steps)
        idle_next = self.next_serverless_idle_step(step, defer_idle=defer_idle)
        if idle_next is not None and idle_next > step:
            candidates.append(idle_next)
        elif idle_next is not None and idle_next <= step:
            candidates.append(step)
        return candidates

    def advance_indexed_busy_work(self, step, skip_steps):
        if skip_steps <= 0:
            return
        # SETUP/SHUTDOWN timers advance with the global pass counter.
        self.timer_passes += skip_steps
        for instance in tuple(self.busy_instances):
            status = instance.getStatus()
            if status not in (Status.ACTIVE, Status.WORKING):
                continue
            busy_slots = 0
            for req in instance.exec_queue:
                if req is None:
                    continue
                if req.workload > 0:
                    req.workload -= instance.processing_capacity * skip_steps
                    busy_slots += 1
            if busy_slots > 0:
                instance.CpuCtr += busy_slots * skip_steps
                if Flg.is_serverless(self.mode) and hasattr(instance, "setLastTime"):
                    # setLastTime notifies the idle heap; refresh after.
                    instance.setLastTime(step + skip_steps - 1)
            self.refresh_instance_schedule(instance)

    def scaleOut(self, metrics):
        # Walk ids ascending — same order as a dense instances[0..N) scan.
        ideal_num_instance = math.ceil(self.num_active_instance * metrics)
        num_active_instance = self.num_active_instance
        for inst_id in range(self.pool_capacity):
            if ideal_num_instance <= num_active_instance:
                return
            instance = self._slots[inst_id]
            if instance is None:
                # Free/INACTIVE slot: activate this id (ascending order).
                instance = self.materialize_instance(
                    inst_id, Status.INACTIVE, from_free=True
                )
                instance.deactivatetimer = -1
                instance.activateInstance(self)
                num_active_instance += 1
                continue

            status = instance.getStatus()
            if status == Status.SETUP:
                num_active_instance += 1
            elif (status == Status.WORKING or status == Status.ACTIVE) and instance.deactivatetimer >= 0:
                instance.deactivatetimer = -1
                self.registerInstance2Balancer(instance)
                num_active_instance += 1
            elif status == Status.INACTIVE or status == Status.SHUTDOWN:
                instance.deactivatetimer = -1
                instance.activateInstance(self)
                num_active_instance += 1
        return

    def hasFreeSlot(self):
        """True if coldStartInstance() could materialize a new instance now."""
        return bool(self._free_set)

    def coldStartInstance(self):
        # Equivalent to "first INACTIVE in ascending id order".
        inst_id = self._take_free_id()
        if inst_id is None:
            return None
        instance = self.materialize_instance(inst_id, Status.INACTIVE, from_free=False)
        instance.activateInstance(self)
        self.registerInstance2Balancer(instance)
        return instance

    def scaleIn(self, metrics):
        ideal_num_instance = max(1, math.ceil(self.num_active_instance * metrics))
        num_active_instance = self.num_active_instance
        for inst_id in range(self.pool_capacity - 1, -1, -1):
            if ideal_num_instance >= num_active_instance:
                return
            instance = self._slots[inst_id]
            if instance is None:
                continue
            status = instance.getStatus()
            if status == Status.WORKING or status == Status.ACTIVE:
                if instance.deactivatetimer < 0:
                    instance.deactivateInstance()
                    self.removeInstanceFromBalancer(instance)
                # Event-driven stepping skips idle ACTIVE instances. Force one
                # step so processRequests flips it to SHUTDOWN and the shutdown
                # countdown starts (same as deactivateOldInstance for serverless).
                self.force_step_instances.add(instance)
                num_active_instance -= 1
            elif status == Status.SETUP:
                instance.deactivateInstance()
                instance.setStatus(Status.SHUTDOWN)
                instance.setuptimer = -1
                num_active_instance -= 1
        return

    def deactivateOldInstance(self, instance):
        if instance.getStatus() != Status.ACTIVE:
            return
        # Already scaling in (timer armed) — do not remove from balancer twice.
        if instance.deactivatetimer >= 0:
            self.force_step_instances.add(instance)
            return
        instance.deactivateInstance()
        self.removeInstanceFromBalancer(instance)
        # Keep original same-step semantics: ACTIVE runStep must still run to
        # flip into SHUTDOWN via processRequests (shutdown countdown starts next).
        self.force_step_instances.add(instance)

    def getMetrics(self):
        self.num_active_instance = 0
        cpu_util = 0
        for instance in self.getRunnableInstances():
            status = instance.getStatus()
            if status == Status.WORKING or status == Status.ACTIVE:
                cpu_util += instance.getCpuUtilization()
                instance.setCpuCtr(0)
                self.num_active_instance += 1
        if self.num_active_instance == 0:
            return 1.0
        cpu_util /= self.num_active_instance
        ideal_instance = cpu_util / self.config.CONFIG_SCALE_TARGET
        return ideal_instance

    def isTime2Check(self, timestep):
        return timestep != 0 and (timestep % (self.config.CONFIG_SCALE_INTERVAL * self.config.SIM_STEP_PER_TIME)) == 0

    def containerScaleCheckNeeded(self):
        """
        True if the next periodic container scale sample could change state.

        When a single ACTIVE instance has been idle since the last sample (CpuCtr=0,
        no queued work), metrics would be ~0 and scaleIn/Out would not fire — the
        periodic check can be skipped until load or instance transitions resume.
        """
        if self.mode != Flg.FLG_CONTAINER:
            return True

        balancer = self.getBalancer()
        if balancer.getRequests():
            return True

        # Prefer indexed sets when available; fall back is not needed after rebuild.
        if self.setup_instances or self.shutdown_instances:
            return True

        active_or_working = 0
        for instance in self.getRunnableInstances():
            status = instance.getStatus()
            if status in (Status.SETUP, Status.SHUTDOWN):
                return True
            if status not in (Status.ACTIVE, Status.WORKING):
                continue
            active_or_working += 1
            if instance.getCpuCtr() > 0:
                return True
            if instance.getQueueLength() > 0:
                return True
            for req in instance.exec_queue:
                if req is not None and req.workload > 0:
                    return True

        if active_or_working == 0:
            return True
        if active_or_working > 1:
            # Idle but over-provisioned: one check may scale in.
            return True
        return False
