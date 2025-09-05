import numpy as np
import numpy.typing as npt
from typing import Tuple
from layers import sigmoid

class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self,
                x: npt.NDArray[np.float64],
                h_prev : npt.NDArray[np.float64],
                c_prev: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        forget = A[:, :H]
        learn = A[:,H:2*H]
        input = A[:, 2*H:3*H]
        output = A[:,3*H:]

        forget = sigmoid(forget)
        learn = np.tanh(learn)
        input = sigmoid(input)
        output = sigmoid(output)

        c_next = forget * c_prev + learn * input
        h_next = output * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, input, forget, learn, output, c_next)
        return h_next, c_next
    
    def backward(self,
                dh_next,
                dc_next):
        
        Wx, Wh, b = self.params
        x, h_prev, c_prev, input, forget, learn, output, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * output) * (1 - tanh_c_next ** 2)

        dc_prev = ds * forget

        din = ds * learn
        dfor = ds * c_prev
        dout = dh_next * tanh_c_next
        dlrn = ds * input

        din *= input * (1 - input)
        dfor *= forget * (1 - forget)
        dout *= output * (1 - output)
        dlrn *= (1 - learn ** 2)

        dA = np.hstack((dfor, dlrn, din, dout))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev
    
class TimeLSTM:
    def __init__(self, Wx: npt.NDArray[np.float64], Wh: npt.NDArray[np.float64], b: npt.NDArray[np.float64], stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs: npt.NDArray[np.float64]):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H) , dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs
    
    def backward(self, dhs: npt.NDArray[np.float64]):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]

        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        
        return dxs
    
    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None



