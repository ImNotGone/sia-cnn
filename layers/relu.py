import numpy as np
from numpy import ndarray
from typing import Tuple

from layers.utils import ForwardPropNotDoneError, activation_functions
from layers.layer import Layer

class Relu(Layer):
    def __init__(self, input_shape: Tuple[int, int, int]):
        self.input_shape = input_shape
        self.out_shape = input_shape

        # cache for back_prop
        self.last_input = None

    def get_output_shape(self):
        return self.output_shape

    def forward_prop(self, input: ndarray):
        if (input.shape != self.input_shape):
            print(f"actual_input_shape: {input.shape}")
            print(f"expected_input_shape: {self.input_shape}")
            raise "input shape specified does not match"

        self.last_input = input
        output = np.where(input > 0, input, 0)
        return output

    def back_prop(self, loss_gradient: ndarray):
        if (self.last_input is None):
            raise ForwardPropNotDoneError
        
        aux = np.where(self.last_input > 0, 1, 0)
        aux = aux * loss_gradient

        self.last_input = None
        return aux