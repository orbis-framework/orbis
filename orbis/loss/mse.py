from orbis.tensor.ops import add, mul
from orbis.tensor.tensor import Tensor

def mse(y_pred, y_true):
    diff = add(y_pred, Tensor(-y_true.value))
    return mul(diff, diff)
