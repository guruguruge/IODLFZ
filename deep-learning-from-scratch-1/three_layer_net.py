import numpy as np
from layers import *
from collections import OrderedDict

class ThreeLayerNet:

    def __init__(self, 
                input_size,
                hidden_size1,
                hidden_size2,
                output_size,
                weight_init_size=0.01
                ):
        self.params = {}
        self.params["W1"] = weight_init_size * np.random.randn(input_size,hidden_size1)
        self.params["b1"] = np.zeros(hidden_size1)
        self.params["W2"] = weight_init_size * np.random.randn(hidden_size1,hidden_size2)
        self.params["b2"] = np.zeros(hidden_size2)
        self.params["W3"] = weight_init_size * np.random.randn(hidden_size2,output_size)
        self.params["b3"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine_l(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu_l()
        self.layers["Affine2"] = Affine_l(self.params["W2"], self.params["b2"])
        self.layers["Relu2"] = Relu_l()
        self.layers["Affine3"] = Affine_l(self.params["W3"], self.params["b3"])
        self.LastLayer = SoftmaxLoss_l()

    def predict(self, x):
        for layer in self.layers.values():
            
