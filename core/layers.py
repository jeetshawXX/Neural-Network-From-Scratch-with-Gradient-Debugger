from core.tensor import Tensor
import random


def matmul(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


class Linear:
    def __init__(self, in_features, out_features):
        # initialize small random weights
        self.W = Tensor(
            [[random.uniform(-0.1, 0.1) for _ in range(out_features)]
             for _ in range(in_features)],
            requires_grad=True
        )

        self.b = Tensor(
            [[0 for _ in range(out_features)]],
            requires_grad=True
        )

        self.input = None

    def forward(self, x: Tensor):
        self.input = x
        out = matmul(x.data, self.W.data)

        # add bias
        for i in range(len(out)):
            for j in range(len(out[0])):
                out[i][j] += self.b.data[0][j]

        return Tensor(out, requires_grad=True)

    def zero_grad(self):
        self.W.zero_grad()
        self.b.zero_grad()

    def update(self, lr):
        for i in range(len(self.W.data)):
            for j in range(len(self.W.data[0])):
                self.W.data[i][j] -= lr * self.W.grad[i][j]

        for j in range(len(self.b.data[0])):
            self.b.data[0][j] -= lr * self.b.grad[0][j]


#######  Backward pass for Linear layer ###########  

def backward(self, grad_output):
    # initialize grads
    self.W.zero_grad()
    self.b.zero_grad()

    # compute dW
    self.W.grad = [[0 for _ in row] for row in self.W.data]

    for i in range(len(self.W.data)):
        for j in range(len(self.W.data[0])):
            for k in range(len(grad_output.data)):
                self.W.grad[i][j] += (
                    self.input.data[k][i] * grad_output.data[k][j]
                )

    # compute db
    self.b.grad = [[0 for _ in self.b.data[0]]]

    for j in range(len(self.b.data[0])):
        for i in range(len(grad_output.data)):
            self.b.grad[0][j] += grad_output.data[i][j]

    # compute grad_input
    grad_input = [[0 for _ in range(len(self.W.data))]
                  for _ in range(len(grad_output.data))]

    for i in range(len(grad_output.data)):
        for j in range(len(self.W.data)):
            for k in range(len(self.W.data[0])):
                grad_input[i][j] += (
                    grad_output.data[i][k] * self.W.data[j][k]
                )

    return Tensor(grad_input)