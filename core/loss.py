from core.tensor import Tensor

class MSELoss:
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred: Tensor, target: Tensor):
        self.pred = pred
        self.target = target

        loss = 0
        for i in range(len(pred.data)):
            for j in range(len(pred.data[0])):
                loss += (pred.data[i][j] - target.data[i][j]) ** 2

        return loss

    def backward(self):
        grad = [[0 for _ in row] for row in self.pred.data]

        for i in range(len(self.pred.data)):
            for j in range(len(self.pred.data[0])):
                grad[i][j] = 2 * (self.pred.data[i][j] - self.target.data[i][j])

        return Tensor(grad)