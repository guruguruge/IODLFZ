import numpy as np


class Relu_l:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Sigmoid_l:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine_l:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        # x = x.reshape(x.shape[0] - 1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)
        return dx


def softmax(a):
    if a.ndim == 2:
        safety = np.max(a, axis=1, keepdims=True)
        exp_a = np.exp(a - safety)
        exp_sum = np.sum(exp_a, axis=1, keepdims=True)
        y = exp_a / exp_sum
    else:
        safety = np.max(a)
        exp_a = np.exp(a - safety)
        exp_sum = np.sum(exp_a)
        y = exp_a / exp_sum
    return y


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    safety = 1e-7
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + safety)) / batch_size


class SoftmaxLoss_l:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax output

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # if t_data is one hot vector
            dx = (self.y - self.t) / batch_size
        else:  #
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
