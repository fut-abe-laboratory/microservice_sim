from request import Request
from instance import Instance
from status import Status
from config import Config
from sim_flg import Flg
import heapq
import math

class Balancer:

    def __init__(self, config):
        self.config = config
        self.instances: list[Instance] = []
        self.requests: list[Request] = []
        self._last_instance_idx = -1
        self._last_instance_obj = None
        self.mode = self.config.CONFIG_INSTANCE_FLG
        self.scaler = None
        # hottest() index: lazy max-heap of (-last_time, reg_seq, push_no, inst).
        # reg_seq follows the order of self.instances, so ties on last_time
        # resolve to the earliest-listed instance exactly like the linear scan.
        self._reg_seq = 0
        self._hot_push = 0
        self._hot_heap = []
        return
    
    @property
    def last_instance(self):
        """Index of the last forwarded-to instance (only roundRobin reads it).

        Resolved lazily: forward2Instance used to call list.index() on every
        assignment, which is O(N) with tens of thousands of instances.
        """
        obj = self._last_instance_obj
        if obj is not None:
            try:
                return self.instances.index(obj)
            except ValueError:
                pass
        return self._last_instance_idx

    @last_instance.setter
    def last_instance(self, value):
        self._last_instance_idx = value
        self._last_instance_obj = None

    def setScaler(self, scaler):
        self.scaler = scaler
        return
        
    def getScaler(self):
        return self.scaler
    
    def addRequest(self, request: Request):
        request.setStatus(Status.UNASSIGNED)
        self.requests.append(request)
        return

    def getRequest(self, i: int):
        return self.requests(i)
    
    def getRequests(self):
        return self.requests

    def clearRequests(self):
        self.requests.clear()
        return
    
    def addInstance(self, instance):
        self.instances.append(instance)
        if getattr(instance, "_bal_seq", None) is None:
            instance._bal_seq = self._reg_seq
            self._reg_seq += 1
        self.note_hot_candidate(instance)
        return
    
    def delInstance(self, instance):
        try:
            self.instances.remove(instance)
        except ValueError:
            # Already unregistered (e.g. duplicate idle-timeout deactivation).
            pass
        else:
            if instance not in self.instances:
                instance._bal_seq = None
        return

    # --- hottest() index ----------------------------------------------------
    def _uses_hot_index(self):
        return self.mode in (Flg.FLG_SERVERLESS, Flg.FLG_SERVERLESS_WARM_WAIT)

    def note_hot_candidate(self, instance):
        """Push instance if it is currently eligible for hottest()."""
        if not self._uses_hot_index():
            return
        seq = getattr(instance, "_bal_seq", None)
        if seq is None or instance.status != Status.ACTIVE or instance.queue:
            return
        last_time = instance.getLastTime()
        self._hot_push += 1
        heapq.heappush(self._hot_heap, (-last_time, seq, self._hot_push, instance))
        if len(self._hot_heap) > 4 * len(self.instances) + 4096:
            self.rebuild_hot_index(reassign=False)

    def rebuild_hot_index(self, reassign=True):
        """Rebuild from self.instances (after checkpoint restore or compaction)."""
        if reassign:
            for instance in self.instances:
                instance._bal_seq = None
            self._reg_seq = 0
            for instance in self.instances:
                if getattr(instance, "_bal_seq", None) is None:
                    instance._bal_seq = self._reg_seq
                    self._reg_seq += 1
        self._hot_heap = []
        self._hot_push = 0
        if not self._uses_hot_index():
            return
        entries = []
        for instance in self.instances:
            if instance.status == Status.ACTIVE and not instance.queue:
                self._hot_push += 1
                entries.append((-instance.getLastTime(), instance._bal_seq, self._hot_push, instance))
        heapq.heapify(entries)
        self._hot_heap = entries
    
    def getInstance(self, i: int):
        return self.instances[i]
    
    def getInstances(self):
        return self.instances
    
    def getNumOfInstances(self):
        return len(self.instances)
    
    def clearInstances(self):
        self.instances.clear()
        return

    def isContainedInstance(self, instance):
        return instance in self.instances

    def runStep(self):
        self.manageQueue()
        return

    def manageQueue(self):
        reqs = self.getRequests()
        remain_reqs = []
        i = 0
        n = len(reqs)
        while i < n:
            req = reqs[i]
            i += 1
            # 1-based position among still-unassigned requests (FIFO front = 1).
            queue_position = len(remain_reqs) + 1
            if self.tryForward2Instance(req, queue_len=queue_position) is None:
                remain_reqs.append(req)
                if self._noFurtherAssignmentThisStep():
                    # Nothing changed by the failed attempt, so every later
                    # request would fail the same way; keep them queued.
                    remain_reqs.extend(reqs[i:])
                    break
                skip = self._warmWaitPositionsStillWaiting(queue_position)
                if skip > 0:
                    remain_reqs.extend(reqs[i:i + skip])
                    i += skip
        reqs[:] = remain_reqs
        return

    def _warmWaitPositionsStillWaiting(self, queue_position):
        """Warm-wait chose to wait at `queue_position`: how many next requests also wait?

        A failed chooseInstance() changes no state, so hottest() and
        idleSetupInstance() stay None and the warm capacity W is unchanged.
        The next requests get positions p+1, p+2, ... and wait exactly while
        p+j <= S * W * mu (see shouldColdStartRatherThanWait).
        """
        if self.mode != Flg.FLG_SERVERLESS_WARM_WAIT:
            return 0
        warm_cpus = self.countWarmCapacity()
        mu = float(self.config.CONFIG_MU)
        if warm_cpus <= 0 or mu <= 0:
            return 0
        absorb = float(self.config.CONFIG_DEFAULT_SETUPTIME) * warm_cpus * mu
        return max(0, math.floor(absorb) - max(int(queue_position), 1))

    def _noFurtherAssignmentThisStep(self):
        """After chooseInstance() returned None: can a later request still be placed?

        Serverless returns None only when no warm instance is idle and the pool
        has no free slot. Warm-wait returns None either by choice (position too
        early to cold-start) or because the pool is full; only the latter holds
        for every later position.
        """
        if self.mode == Flg.FLG_SERVERLESS:
            return True
        if self.mode == Flg.FLG_SERVERLESS_WARM_WAIT:
            return not self.getScaler().hasFreeSlot()
        return False
    
    def tryForward2Instance(self, req, queue_len=None):
        instance = self.chooseInstance(queue_len=queue_len)
        if instance:
            return self.forward2Instance(instance, req)
        else:
            return None

    def chooseInstance(self, queue_len=None):
        if queue_len is None:
            queue_len = len(self.getRequests())

        if self.mode == Flg.FLG_CONTAINER:
            # round robin
            # instance = self.roundRobin()
            
            # 0916 balancer with queue
            for instance in self.getInstances():
                if (instance.getStatus() == Status.ACTIVE or instance.getStatus() == Status.WORKING) and instance.getQueueLength() == 0:
                    return instance
            return None

        elif self.mode == Flg.FLG_SERVERLESS:
            # hottest
            instance = self.hottest()
            if instance == None:
                instance = self.coldStart()
            return instance

        elif self.mode == Flg.FLG_SERVERLESS_WARM_WAIT:
            instance = self.hottest()
            if instance is not None:
                return instance
            # Prefer an already-starting instance over opening another cold start.
            setup = self.idleSetupInstance()
            if setup is not None:
                return setup
            if self.shouldColdStartRatherThanWait(queue_len=queue_len):
                return self.coldStart()
            return None

        return None

    def idleSetupInstance(self):
        """SETUP instance with empty queue (capacity reserved for one waiting request)."""
        for instance in self.getInstances():
            if instance.getStatus() == Status.SETUP and instance.getQueueLength() == 0:
                return instance
        return None

    def countWarmCapacity(self):
        """Number of warm (ACTIVE/WORKING) CPU slots currently registered to the balancer."""
        warm_cpus = 0
        for instance in self.getInstances():
            status = instance.getStatus()
            if status == Status.ACTIVE or status == Status.WORKING:
                warm_cpus += instance.getNumCPU()
        return warm_cpus

    def expectedWarmWaitTime(self, queue_position):
        """
        Expected wait (logical time) for the request at 1-based position
        `queue_position` in the balancer queue: p / (warm_cpus * mu).
        """
        warm_cpus = self.countWarmCapacity()
        mu = float(self.config.CONFIG_MU)
        if warm_cpus <= 0 or mu <= 0:
            return float("inf")
        p = max(int(queue_position), 1)
        return p / (warm_cpus * mu)

    def shouldColdStartRatherThanWait(self, queue_len):
        """
        Cold-start only when THIS request's expected warm-wait exceeds setup time.

        With warm capacity W (CPU slots), setup time S, service rate μ:
          absorb ≈ S * W * μ   # requests warm pool can drain within S
          cold-start iff queue_position > absorb

        `queue_len` here is the 1-based position of the current request among
        still-unassigned balancer requests (earlier deferred + current).
        """
        setup_time = float(self.config.CONFIG_DEFAULT_SETUPTIME)
        warm_cpus = self.countWarmCapacity()
        mu = float(self.config.CONFIG_MU)
        position = max(int(queue_len), 1)

        if warm_cpus <= 0 or mu <= 0:
            # No warm capacity: open a new instance (idle SETUP handled earlier).
            return True

        absorb = setup_time * warm_cpus * mu
        return position > absorb

    def hasAssignableOrScalableWork(self):
        """True if balancer queue can be progressed this step (assign or cold-start)."""
        if not self.getRequests():
            return False
        if self.mode == Flg.FLG_CONTAINER:
            # Container dispatch only targets ACTIVE/WORKING with an empty queue.
            return self.chooseInstance() is not None
        if self.mode == Flg.FLG_SERVERLESS:
            # hottest() or a cold start; with a full pool and no idle warm
            # instance nothing can change until a setup / completion / shutdown.
            return self.getScaler().hasFreeSlot() or self.hottest() is not None
        if self.mode != Flg.FLG_SERVERLESS_WARM_WAIT:
            return True
        if self.hottest() is not None:
            return True
        if self.idleSetupInstance() is not None:
            return True
        if not self.getScaler().hasFreeSlot():
            return False
        return self.shouldColdStartRatherThanWait(queue_len=1)
    
    def roundRobin(self):
        instance = None
        if self.last_instance + 1 < len(self.getInstances()):
            next_instance = self.last_instance + 1
        else:
            next_instance = 0
        instance = self.getInstance(next_instance)

        return instance
    
    def hottest(self):
        if self._uses_hot_index():
            heap = self._hot_heap
            while heap:
                neg_last, seq, _, instance = heap[0]
                if (
                    getattr(instance, "_bal_seq", None) == seq
                    and instance.status == Status.ACTIVE
                    and not instance.queue
                    and instance.getLastTime() == -neg_last
                ):
                    return instance
                heapq.heappop(heap)
            return None
        return self._hottest_scan()

    def _hottest_scan(self):
        # Most recently used idle ACTIVE instance; ties -> earliest in list.
        # Staggered starters have last_time < 0, so no -1 sentinel cut-off
        # (it used to make every starter except the first unreachable).
        hottest = None
        last_time = None
        for instance in self.getInstances():
            if instance.getStatus() == Status.ACTIVE and instance.getQueueLength() == 0:
                temp_last = instance.getLastTime()
                if hottest is None or last_time < temp_last:
                    last_time = temp_last
                    hottest = instance
        return hottest
    
    def coldStart(self):
        instance = None
        scaler = self.getScaler()
        instance = scaler.coldStartInstance()
        return instance

    def forward2Instance(self, instance: Instance, req: Request):
        # self.delRequest(req)
        if self.mode == Flg.FLG_CONTAINER:
            req.setProcessedBy("container" + str(instance.getId()))
        elif self.mode == Flg.FLG_SERVERLESS:
            req.setProcessedBy("serverless" + str(instance.getId()))
        elif self.mode == Flg.FLG_SERVERLESS_WARM_WAIT:
            req.setProcessedBy("serverless_warm_wait" + str(instance.getId()))
        instance.addRequest(req)
        self._last_instance_obj = instance
        return req

    def delRequest(self, req:Request):
        self.requests.remove(req)
        return
