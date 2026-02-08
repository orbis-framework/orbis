class Tensor:
    def __init__(self, value, requires_grad=False):
        self.value = float(value)
        self.requires_grad = requires_grad
        self.grad = 0.0
        self._backward = lambda: None
        self._parents = []

    def backward(self):
        self.grad = 1.0
        topo = []
        visited = set()

        def build(t):
            if t not in visited:
                visited.add(t)
                for p in t._parents:
                    build(p)
                topo.append(t)

        build(self)
        for t in reversed(topo):
            t._backward()
