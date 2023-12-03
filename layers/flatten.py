from numpy import ndarray
import numpy as np

from layers.layer import Layer


class Flatten(Layer):
    def initialize(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape[0] * input_shape[1] * input_shape[2]

    def get_output_shape(self):
        return self.output_shape

    def forward_prop(self, input: ndarray):
        self.input_shape = input.shape
        return input.flatten()

    def back_prop(self, loss_gradient: ndarray):
        return loss_gradient.reshape(self.input_shape)
