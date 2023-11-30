from numpy import ndarray
import numpy as np

from layers.layer import Layer

class Flatten(Layer):

    def forward_prop(self, input: ndarray):
        self.input_shape = input.shape
        self.output_shape = (input.shape[0], np.prod(input.shape[1:]))
        return input.reshape(self.output_shape)

    def back_prop(self, loss_gradient: ndarray):
        return loss_gradient.reshape(self.input_shape)

