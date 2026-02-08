from orbis.core.optype import OpType
import random

class CPUBackend:
    available = True
    capabilities = {OpType.DETERMINISTIC}

    def run(self, op):
        return op.payload()

class ProbabilisticBackend:
    available = True
    capabilities = {OpType.PROBABILISTIC}

    def run(self, op):
        return op.payload()

class QPUBackend:
    def __init__(self, available=False):
        self.available = available
        self.capabilities = {OpType.QUANTUM}

    def run(self, op):
        return random.choice([0, 1])
