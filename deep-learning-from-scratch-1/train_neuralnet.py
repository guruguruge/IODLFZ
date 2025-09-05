import numpy as np
from mnist import load_mnist


# sigmoid function : for activating neurons, in hiden layers
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sofrmax function : for finalizing output
def softmax(a):
    safety = np.max(a)
    exp_a = np.exp(a - safety)
    exp_sum = np.sum(exp_a)
    y = exp_a / exp_sum
    return y


# prediciton : y , teacher : t
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    safety = 1e-7
    batch_size = y.shape[0]
    return np.sum(t * np.log(y + safety)) / batch_size


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    # for idx in range(x.size):
    # for multi index 784 * minibatch
    while not it.finished:
        idx = it.multi_index
        # temporarly claculate grdient in each index and making grad function and putting x back
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        tmp_val = x[idx]
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

        it.iternext

    return grad


# machine learning by gradient deceent
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # init of weights and bias
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        # making predictions with matrix calculation
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # claculating error
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}

        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads
