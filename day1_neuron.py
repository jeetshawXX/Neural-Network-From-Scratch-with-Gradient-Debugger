def matmul(A, B):
    result = [[0 for j in range(len(B[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


def add_bias(X, b):
    return [[X[i][j] + b[j] for j in range(len(b))] for i in range(len(X))]

def relu(X):
    return [[max(0, x) for x in row] for row in X]


x = [[20, 10]]

#weights(2 inputs -> 1 neuron)
w = [[0.5], [-0.3]]

#bias
b = [0.1]

#forward pass
z = matmul(x, w)

z = add_bias(z, b)
output = relu(z)

print("Input:", x)
print("Weights:", w)
print("Bias:", b)
print("Output:", output)


y_true = [[10]]

#loss function
def mse(y_pred, y_true):
    loss = 0
    for i in range(len(y_pred)):
        for j in range(len(y_pred[0])):
            loss += (y_pred[i][j] - y_true[i][j]) ** 2
    return loss

loss = mse(output, y_true)
print("Loss:", loss)


# Backward pass

# dL/dy_pred (derivative of MSE)
dL_dy = 2 * (output[0][0] - y_true[0][0])

# ReLU derivative (since z > 0 → derivative = 1)
dy_dz = 1 if z[0][0] > 0 else 0

dL_dz = dL_dy * dy_dz

# Gradients w.r.t weights
dL_dw0 = dL_dz * x[0][0]
dL_dw1 = dL_dz * x[0][1]

# Gradient w.r.t bias
dL_db = dL_dz

lr = 0.001

w[0][0] -= lr * dL_dw0
w[1][0] -= lr * dL_dw1
b[0] -= lr * dL_db

for epoch in range(1000):

    # forward
    z = matmul(x, w)
    z = add_bias(z, b)
    output = relu(z)

    # loss
    loss = mse(output, y_true)

    # backward
    dL_dy = 2 * (output[0][0] - y_true[0][0])
    dy_dz = 1 if z[0][0] > 0 else 0
    dL_dz = dL_dy * dy_dz

    dL_dw0 = dL_dz * x[0][0]
    dL_dw1 = dL_dz * x[0][1]
    dL_db = dL_dz

    # update
    lr = 0.001
    w[0][0] -= lr * dL_dw0
    w[1][0] -= lr * dL_dw1
    b[0] -= lr * dL_db

    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", loss)