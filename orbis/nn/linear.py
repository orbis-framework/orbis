from orbis.tensor.tensor import Tensor
from orbis.tensor.ops import add, mul

class Linear:
    def __init__(self, in_features, out_features):
        self.w = Tensor(0.0, requires_grad=True)
        self.b = Tensor(0.0, requires_grad=True)

    def __call__(self, x):
        return add(mul(self.w, x), self.b)

    def parameters(self):
        return [self.w, self.b]
