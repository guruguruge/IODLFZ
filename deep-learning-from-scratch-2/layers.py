import numpy as np
import numpy.typing as npt
from typing import Tuple, List

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
    

class TimeSoftmaxWithLoss:

    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs : npt.NDArray[np.float64],
                ts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        N, T, V = xs.shape

        if ts.ndim == 3:
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        xs = xs.reshape(N*T , V)
        ts = ts.reshape(N*T)
        mask = mask.reshape(N*T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))

        return loss
    
    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]

        dx = dx.reshape((N, T, V))

        return dx


class RNN:
    def __init__(self, Wx : npt.NDArray[np.float64], Wh: npt.NDArray[np.float64], b: npt.NDArray[np.float64]):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        self.cache = None

    def forward(self, x : npt.NDArray[np.float64], h_prev : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        Wx, Wh, b = self.params

        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next
    
    def backward(self, dh_next : npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        return dx, dh_prev
    
class TimeRNN:
    def __init__(self,
                  Wx : npt.NDArray[np.float64],
                  Wh: npt.NDArray[np.float64],
                  b: npt.NDArray[np.float64],
                  stateful : bool):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h : npt.NDArray[np.float64]):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs
    
    def backward(self, dhs : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        Wx, Wh, b = self.params
        N, T, D = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]

        for  t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh

        return dxs
    
GPU = False
class Embedding:
    def __init__(self, W : npt.NDArray[np.float64]):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx : int) -> npt.NDArray[np.float64]:
        W = self.params[0]
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW = self.grads[0]
        dW[...] = 0

        if GPU:
            np.scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)
        return None
        
class TimeEmbedding:
    def __init__(self, W : npt.NDArray[np.float64]):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        N, T = xs.shape
        V, D = self.W.shape
        
        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out
    
    def backward(self, dout : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None
    
class TimeAffine:
    def __init__(self, W : npt.NDArray[np.float64], b : npt.NDArray[np.float64]):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x =None

    def forward(self, x :npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N*T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)
    
    def backward(self, dout : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx