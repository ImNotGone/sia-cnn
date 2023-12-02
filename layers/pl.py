from enum import Enum
import numpy as np
from numpy import ndarray
from typing import Tuple

from layers.utils import ForwardPropNotDoneError
from layers.layer import Layer


class PoolingType(Enum):
    MAX = np.amax
    MIN = np.amin
    AVG = np.average


class PL(Layer):
    def __init__(self, input_shape: Tuple[int, int, int], pooling_type: PoolingType = PoolingType.MAX, stride: int = 2):
        self.stride = stride
        self.pooling_func = pooling_type

        self.input_shape = input_shape
        self.output_shape = (input_shape[0], input_shape[1] // self.stride, input_shape[2] // self.stride)

        # cache for back_prop
        self.last_input = None

    def get_output_shape(self):
        return self.output_shape

    def iterate_regions(self, input:ndarray):
        channels, heigth, width = input.shape
        heigth = heigth // self.stride
        width  = width  // self.stride

        for i in range(channels):
            for j in range(heigth):
                for k in range(width):
                    j_start = j * self.stride
                    j_end = j * self.stride + self.stride
                    k_start = k * self.stride
                    k_end = k * self.stride + self.stride
                    region = input[i, j_start:j_end, k_start:k_end]
                    yield region, i, j, k
                

    def forward_prop(self, input: ndarray):
        if (input.shape != self.input_shape):
            print(f"actual_input_shape: {input.shape}")
            print(f"expected_input_shape: {self.input_shape}")
            raise "input shape specified does not match"
        
        # cache'd for easier back_prop
        self.last_input = input

        output = np.zeros(self.output_shape)

        for region, i, j, k in self.iterate_regions(input):
            output[i, j, k] = self.pooling_func(region)

        return output

    def back_prop(self, loss_gradient: ndarray):
        # We cached the last input image while doing forward_prop to make back_prop easier
        # We check that forward propagation was done before doing back propagation
        if (self.last_input is None):
            # TODO implement error
            raise ForwardPropNotDoneError

        input = np.zeros(self.last_input.shape)

        for region, i, j, k in self.iterate_regions(input):
            pool_value = self.pooling_func(region)

            for m, n in np.ndindex(region.shape):
                # if the pixel is pool value -> transfer gradient
                if[region[m][n]] == pool_value:
                    m_start = m * self.stride
                    m_end = m * self.stride + self.stride
                    n_start = n * self.stride
                    n_end = n * self.stride + self.stride
                    input[i, m_start:m_end, n_start:n_end] = loss_gradient[i, j, k]

        self.last_input = None

        return input
