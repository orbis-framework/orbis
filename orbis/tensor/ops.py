from orbis.tensor.tensor import Tensor

def add(a, b):
    out = Tensor(a.value + b.value, requires_grad=a.requires_grad or b.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += out.grad
        if b.requires_grad:
            b.grad += out.grad

    out._parents = [a, b]
    out._backward = _backward
    return out

def mul(a, b):
    out = Tensor(a.value * b.value, requires_grad=a.requires_grad or b.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += b.value * out.grad
        if b.requires_grad:
            b.grad += a.value * out.grad

    out._parents = [a, b]
    out._backward = _backward
    return out
