from core.layers import Linear
from core.tensor import Tensor

x = Tensor([[20, 10]])

layer = Linear(2, 1)

out = layer.forward(x)

print("Output:", out)
print("Weights:", layer.W)
print("Bias:", layer.b)
