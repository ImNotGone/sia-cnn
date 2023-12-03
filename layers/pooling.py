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


class Pooling(Layer):
    def __init__(self, input_shape: Tuple[int, int, int], pooling_type: PoolingType = PoolingType.MAX, stride: int = 2):
        self.stride = stride
        self.pooling = pooling_type

        self.input_shape = input_shape
        self.output_shape = (input_shape[0], input_shape[1] // self.stride, input_shape[2] // self.stride)

        # cache for back_prop
        self.last_input = None
        self.last_output = None

    def get_output_shape(self):
        return self.output_shape

    def iterate_regions(self, input:ndarray):
        # output shape has heigth and width already divided by stride
        channels, heigth, width = self.output_shape

        for i in range(channels):
            for j in range(heigth):
                for k in range(width):
                    j_start = j * self.stride
                    j_end   = j_start + self.stride
                    k_start = k * self.stride
                    k_end   = k_start + self.stride
                    region  = input[i, j_start:j_end, k_start:k_end]
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
            output[i, j, k] = self.pooling(region)

        # cache'd for easier back_prop
        self.last_output = output
        return output
   
    # Loads loss_gradient to the value that was pooled on foward prop
    def _single_value_back_prop(self, loss_gradient: ndarray):
        gradient = np.zeros(self.last_input.shape)
        for i, j, k in np.ndindex(gradient.shape):
            if(self.last_input[i][j][k] == self.last_output[i][j//self.stride][k//self.stride]):
                gradient[i][j][k] = loss_gradient[i][j//self.stride][k//self.stride]
        return gradient

    # distributes loss_gradient on all stride * stride matrices
    def _avg_back_prop(self, loss_gradient: ndarray):
        return loss_gradient.repeat(self.stride, axis=1).repeat(self.stride,axis=2) / (self.stride**2)
    
    def back_prop(self, loss_gradient: ndarray):
        # We cached the last input while doing forward_prop to make back_prop easier
        # We check that forward propagation was done before doing back propagation
        if (self.last_input is None):
            raise ForwardPropNotDoneError

        ret = None
        match(self.pooling):
            case(PoolingType.MIN):
                ret = self._single_value_back_prop(loss_gradient)
            case(PoolingType.MAX):
                ret =  self._single_value_back_prop(loss_gradient)
            case(PoolingType.AVG):
                ret =  self._avg_back_prop(loss_gradient)
        
        if (ret is None):
            raise "Backprop for current pooling not found!"
        
        # so that the guard clause works
        self.last_input = None
        return ret
