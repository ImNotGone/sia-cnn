import numpy as np
from typing import Tuple
from numpy import ndarray

from layers.utils import ForwardPropNotDoneError
from layers.layer import Layer


class Relu(Layer):
    def __init__(self):
        # cache for back_prop
        self.last_input = None

    def initialize(self, input_shape: Tuple[int, int, int]):
        self.input_shape = input_shape
        self.output_shape = input_shape

    def get_output_shape(self):
        return self.output_shape

    def forward_prop(self, input: ndarray):
        self.last_input = input
        output = np.where(input > 0, input, 0)
        return output

    def back_prop(self, loss_gradient: ndarray):
        if self.last_input is None:
            raise ForwardPropNotDoneError

        # if last_input has a value > 0 -> return the loss_gradient for that value since derivative is 1
        aux = np.where(self.last_input > 0, loss_gradient, 0)

        self.last_input = None
        return aux
