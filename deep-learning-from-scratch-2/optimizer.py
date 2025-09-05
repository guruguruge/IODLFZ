import numpy as np
from typing import List, Tuple


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params : List, grad : List):

        for i in range(len(params)):
            params[i] -= self.lr * grad[i]


class Momentum:
    def __init__(self, ac=0.9, lr=0.01):
        self.lr = lr
        self.ac = ac
        self.velo = None

    def update(self, params, grad):
        if self.velo is None:
            self.velo = {}
            for key, val in params.items():
                self.velo[key] = np.zeros_like(val)

        for key in params:
            self.velo[key] = self.ac * self.velo[key] - self.lr * grad[key]
            params[key] -= self.velo[key]


class AddGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grad):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params:
            self.h[key] += grad[key] * grad[key]
            params[key] -= self.lr * grad[key] / np.sqrt(self.h[key] + 1e-7)
