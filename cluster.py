import numpy.random as rd
import math
from request import Request
from instance import Instance
from limit import Limit
from config import Config
from status import Status

class Cluster:

    def __init__(self, config):
        self.config = config
        self.balancer = None
        self.instances = []
        self.scaler = None
        self.num_CPU = self.config.CONFIG_CLUSTER_CPU
        self.num_active_CPU = 0
        self.pool_capacity = 0
        return

    def reservePool(self, capacity: int, cpu_per_instance: int):
        """Reserve cluster CPU for a fixed instance pool without materializing."""
        capacity = int(capacity)
        cpu_per_instance = int(cpu_per_instance)
        needed = capacity * cpu_per_instance
        if needed > self.getMaxCPU():
            print("ERROR: shortage CPU")
            return False
        self.pool_capacity = capacity
        self.setActiveCPU(needed)
        return True

    def addBalancer(self, balancer):
        self.balancer = balancer
        return

    def getBalancer(self):
        return self.balancer
    
    def getActiveCPU(self):
        return self.num_active_CPU
    
    def setActiveCPU(self, cpu):
        self.num_active_CPU = cpu

    def getMaxCPU(self):
        return self.num_CPU

    def addInstance(self, instance):
        num_cpu = instance.getNumCPU() + self.getActiveCPU()
        if num_cpu > self.getMaxCPU():
            print("ERROR: shortage CPU")
            return False
        else:
            self.instances.append(instance)
            self.setActiveCPU(num_cpu) 
        return True

    def getInstances(self):
        return self.instances
    
    def getNumOfInstances(self):
        # Report configured pool size (matches pre-lazy dense allocation).
        if self.pool_capacity > 0:
            return int(self.pool_capacity)
        scaler = self.getScaler()
        if scaler is not None and getattr(scaler, "pool_capacity", 0) > 0:
            return int(scaler.pool_capacity)
        return len(self.instances)

    def noteMaterializedInstance(self, instance):
        """Track a lazily created instance without changing reserved CPU."""
        if instance not in self.instances:
            self.instances.append(instance)
        return

    def noteDematerializedInstance(self, instance):
        if instance in self.instances:
            self.instances.remove(instance)
        return

    def addScaler(self, scaler):
        self.scaler = scaler
        return

    def getScaler(self):
        return self.scaler
    
    def registerInstance2Scaler(self, instance):
        self.scaler.addInstance(instance)
        return

    def addRequest(self, request: Request):
        self.getBalancer().addRequest(request)
        return