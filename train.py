from core.tensor import Tensor
from core.layers import Linear
from core.activations import ReLU
from core.loss import MSELoss

# data
x = Tensor([[20, 10]])
y = Tensor([[10]])

# model
layer = Linear(2, 1)
relu = ReLU()
criterion = MSELoss()

lr = 0.001

for epoch in range(1000):

    # forward
    out = layer.forward(x)
    out = relu.forward(out)
    loss = criterion.forward(out, y)

    # backward
    grad_loss = criterion.backward()
    grad_relu = relu.backward(grad_loss)
    layer.backward(grad_relu)

    # update
    layer.update(lr)

    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", loss)

print("Final Weights:", layer.W)
print("Final Bias:", layer.b)



#######  Using MLP model instead of single layer ###########

from core.tensor import Tensor
from core.loss import MSELoss
from models.mlp import MLP

x = Tensor([[20, 10]])
y = Tensor([[10]])

model = MLP()
criterion = MSELoss()

lr = 0.001

for epoch in range(2000):

    # forward
    out = model.forward(x)
    loss = criterion.forward(out, y)

    # backward
    grad_loss = criterion.backward()
    model.backward(grad_loss)

    # update
    model.update(lr)

    if epoch % 200 == 0:
        print("Epoch:", epoch, "Loss:", loss)

print("Final output:", model.forward(x))