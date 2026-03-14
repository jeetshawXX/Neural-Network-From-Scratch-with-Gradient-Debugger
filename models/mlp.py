from core.layers import Linear
from core.activations import ReLU


class MLP:
    def __init__(self):
        self.fc1 = Linear(2, 4)
        self.relu = ReLU()
        self.fc2 = Linear(4, 1)

    def forward(self, x):
        out = self.fc1.forward(x)
        out = self.relu.forward(out)
        out = self.fc2.forward(out)
        return out

    def backward(self, grad_output):
        grad = self.fc2.backward(grad_output)
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad)

    def update(self, lr):
        self.fc1.update(lr)
        self.fc2.update(lr)