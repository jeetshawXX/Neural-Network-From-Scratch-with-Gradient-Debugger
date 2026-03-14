class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None

    def zero_grad(self):
        if self.requires_grad:
            self.grad = [[0 for _ in row] for row in self.data]

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

