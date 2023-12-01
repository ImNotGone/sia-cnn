from numpy import ndarray
import numpy as np

from layers.layer import Layer

class Flatten(Layer):

    def forward_prop(self, input: ndarray):
        self.input_shape = input.shape
        return input.flatten()

    def back_prop(self, loss_gradient: ndarray):
        return loss_gradient.reshape(self.input_shape)

