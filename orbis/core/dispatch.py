from orbis.core.optype import OpType

class DispatchResult:
    def __init__(self, value, backend, fallback, optype):
        self.value = value
        self.backend = backend
        self.fallback = fallback
        self.optype = optype

    def __repr__(self):
        return (
            f"{self.value}\n"
            f"  backend={self.backend}, "
            f"fallback={self.fallback}, "
            f"type={self.optype}"
        )

class Dispatcher:
    def __init__(self, backends):
        self.backends = backends

    def dispatch(self, op):
        for backend in self.backends:
            if backend.available and op.optype in backend.capabilities:
                return DispatchResult(
                    backend.run(op),
                    backend.__class__.__name__,
                    False,
                    op.optype.value
                )

        if op.optype == OpType.QUANTUM:
            for backend in self.backends:
                if backend.available and OpType.PROBABILISTIC in backend.capabilities:
                    return DispatchResult(
                        backend.run(op),
                        backend.__class__.__name__,
                        True,
                        op.optype.value
                    )

        raise RuntimeError(f"No backend for {op.name}")
