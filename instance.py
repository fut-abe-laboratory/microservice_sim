from request import Request
from status import Status
from config import Config
from collections import deque

class Instance:
    def __init__(self, capacity, num_CPU, queue_length, status, config:Config):
        self.config = config
        self.id = -1
        # Raw timer values. While the instance is in SETUP (setuptimer) or
        # SHUTDOWN (deactivatetimer) under a scaler, the timer is "virtual":
        # it counts down with the scaler's global pass counter instead of being
        # decremented on every step (see Scaler.timer_passes).
        self._setuptimer = -1
        self._deactivatetimer = -1
        self._timer_anchor = None
        self._timer_token = 0
        self.status = status
        self.status_manager = None
        self.processing_capacity = capacity / self.config.SIM_STEP_PER_TIME
        self.num_CPU = num_CPU
        self.max_queue_length = queue_length
        self.queue = deque()
        self.exec_queue = []
        for i in range(self.num_CPU):
            self.exec_queue.append(None)
        self.CpuCtr = 0
        return
    
    def setId(self, id):
        self.id = id

    def getId(self):
        return self.id

    def setCapacity(self, capacity):
        self.processing_capacity = capacity
        return

    def getCapacity(self):
        return self.processing_capacity
    
    def getQueueLength(self):
        return len(self.queue)
    
    def setNumCPU(self, num_CPU):
        self.num_CPU = num_CPU
        return

    def getNumCPU(self):
        return self.num_CPU

    def getMaxQueueLength(self):
        return self.max_queue_length
    
    def addRequest(self, req:Request):
        max_queue_length = self.max_queue_length
        if max_queue_length < 0 or len(self.queue) < max_queue_length:
            req.setStatus(Status.QUEUEING)
            self.queue.append(req)
        else:
            req.setStatus(Status.DROP)
        status_manager = self.status_manager
        if status_manager is not None and hasattr(status_manager, "refresh_instance_schedule"):
            status_manager.refresh_instance_schedule(self)
        return
    
    def delRequest(self, req:Request):
        # req.setStatus(Status.FINISHED)
        self.queue.popleft()
        return
    

    # --- Virtual setup/shutdown timers -------------------------------------
    def _timer_passes(self):
        sm = self.status_manager
        return None if sm is None else getattr(sm, "timer_passes", None)

    def _timer_elapsed(self):
        anchor = self._timer_anchor
        if anchor is None:
            return 0
        passes = self._timer_passes()
        return 0 if passes is None else passes - anchor

    @property
    def setuptimer(self):
        if getattr(self, "status", None) == Status.SETUP and self._timer_anchor is not None:
            return self._setuptimer - self._timer_elapsed()
        return self._setuptimer

    @setuptimer.setter
    def setuptimer(self, value):
        self._setuptimer = value
        if getattr(self, "status", None) == Status.SETUP and self._timer_anchor is not None:
            self._anchor_timer()

    @property
    def deactivatetimer(self):
        if getattr(self, "status", None) == Status.SHUTDOWN and self._timer_anchor is not None:
            return self._deactivatetimer - self._timer_elapsed()
        return self._deactivatetimer

    @deactivatetimer.setter
    def deactivatetimer(self, value):
        self._deactivatetimer = value
        if getattr(self, "status", None) == Status.SHUTDOWN and self._timer_anchor is not None:
            self._anchor_timer()

    def _materialize_timer(self):
        """Fold elapsed passes into the raw value of the running timer."""
        if self._timer_anchor is None:
            return
        elapsed = self._timer_elapsed()
        if self.status == Status.SETUP:
            self._setuptimer -= elapsed
        elif self.status == Status.SHUTDOWN:
            self._deactivatetimer -= elapsed
        self._timer_anchor = None

    def _anchor_timer(self):
        """Start (or restart) the virtual countdown at the current pass."""
        passes = self._timer_passes()
        if passes is None:
            self._timer_anchor = None
            return
        self._timer_anchor = passes
        self._timer_token += 1
        sm = self.status_manager
        if sm is not None and hasattr(sm, "schedule_timer"):
            sm.schedule_timer(self)

    def timer_virtual(self):
        return self._timer_anchor is not None

    def setStatus(self, status):
        prev_status = self.status
        if prev_status != status and prev_status in (Status.SETUP, Status.SHUTDOWN):
            self._materialize_timer()
        self.status = status
        if prev_status != status and status in (Status.SETUP, Status.SHUTDOWN):
            self._anchor_timer()
        status_manager = self.status_manager
        if status_manager is not None and prev_status != status:
            if prev_status == Status.INACTIVE and status != Status.INACTIVE:
                status_manager.registerRunnableInstance(self)
            elif prev_status != Status.INACTIVE and status == Status.INACTIVE:
                status_manager.removeRunnableInstance(self)
            if hasattr(status_manager, "on_instance_status_changed"):
                status_manager.on_instance_status_changed(self, prev_status, status)
        return

    def setStatusManager(self, status_manager):
        self.status_manager = status_manager
        return
    
    def getStatus(self):
        return self.status
    
    def processRequest(self, req:Request, time):
        return None

    def processRequests(self, time):
        end_reqs: list[Request] = []
        num_CPU = self.num_CPU
        queue = self.queue
        exec_queue = self.exec_queue
        is_process = False
        
        for i in range(num_CPU):
            if exec_queue[i] is not None:
                req = exec_queue[i]
                self.setStatus(Status.WORKING)
                return_req = self.processRequest(req, time)
                is_process = True
                if return_req is not None:
                    end_reqs.append(return_req)
                    exec_queue[i] = None
                else:
                    self.incrementCpuCtr()
                    continue
            if exec_queue[i] is None and queue:
                req = queue.popleft()
                exec_queue[i] = req
                
                self.setStatus(Status.WORKING)
                return_req = self.processRequest(req, time)
                is_process = True
                self.incrementCpuCtr()

        
        # for req in self.queue:
        #     if num_CPU > 0:
        #         self.setStatus(Status.WORKING)
        #         return_req = self.processRequest(req, time)
        #         is_process = True
        #         if return_req is not None:
        #             end_reqs.append(return_req)
        #         else:
        #             self.incrementCpuCtr()
        #             num_CPU -= 1
        #     else:
        #         break

        if not is_process:
            if self.deactivatetimer < 0:
                self.setStatus(Status.ACTIVE)
            else:
                self.setStatus(Status.SHUTDOWN)

        return end_reqs, is_process
    
    def getCpuUtilization(self):
        return self.getCpuCtr() / ((self.config.CONFIG_SCALE_INTERVAL * self.config.SIM_STEP_PER_TIME) * self.getNumCPU())
    
    def incrementCpuCtr(self):
        self.CpuCtr += 1

    def setCpuCtr(self, ctr):
        self.CpuCtr = ctr
    
    def getCpuCtr(self):
        return self.CpuCtr
    
    def activateInstance(self, scaler):
        return
    
    def setupInstance(self):
        return
    
    def deactivateInstance(self):
        self.deactivatetimer = self.config.CONFIG_DEFAULT_SHUTDOWNTIME * self.config.SIM_STEP_PER_TIME
        # print("START SCALE IN")
        return
    
    def shutdownInstance(self):
        if not self.timer_virtual():
            self.deactivatetimer -= 1
        if self.deactivatetimer < 0:
            self.setStatus(Status.INACTIVE)
            self.setCpuCtr(0)
            # print("COMPLETE SCALE IN: " + str(self.getId()))
        return
    
    def runStep(self, time):
        status = self.getStatus()
        if status == Status.ACTIVE or status == Status.WORKING:
            end_reqs, is_process = self.processRequests(time)
            status_manager = self.status_manager
            if status_manager is not None and hasattr(status_manager, "refresh_instance_schedule"):
                status_manager.refresh_instance_schedule(self)
            return end_reqs
        elif status == Status.SETUP:
            self.setupInstance()
        elif status == Status.SHUTDOWN:
            self.shutdownInstance()
        return []
