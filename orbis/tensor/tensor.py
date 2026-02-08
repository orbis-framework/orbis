class Tensor:
    def __init__(self, value, requires_grad=False):
        self.value = float(value)
        self.requires_grad = requires_grad
        self.grad = 0.0
        self._backward = lambda: None
        self._parents = []

    def __repr__(self):
        return f"Tensor(value={self.value}, grad={self.grad})"

    def backward(self):
        """
        Backpropagate gradients from this tensor.
        """
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

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.value + other.value)
        out._parents = [self, other]

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.value * other.value)
        out._parents = [self, other]

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self + (-other)

    def __neg__(self):
        out = Tensor(-self.value)
        out._parents = [self]

        def _backward():
            self.grad -= out.grad

        out._backward = _backward
        return out