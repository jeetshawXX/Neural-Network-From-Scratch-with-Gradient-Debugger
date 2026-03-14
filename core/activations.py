from core.tensor import Tensor

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x: Tensor):
        self.input = x
        out = [[max(0, val) for val in row] for row in x.data]
        return Tensor(out, requires_grad=True)

    def backward(self, grad_output: Tensor):
        grad_input = [[0 for _ in row] for row in self.input.data]

        for i in range(len(self.input.data)):
            for j in range(len(self.input.data[0])):
                if self.input.data[i][j] > 0:
                    grad_input[i][j] = grad_output.data[i][j]
                else:
                    grad_input[i][j] = 0

        return Tensor(grad_input)