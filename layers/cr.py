import numpy as np
from enum import Enum
from typing import Tuple
from numpy import ndarray

from layers.utils import ForwardPropNotDoneError
from layers.utils.optimization_methods import OptimizationMethod
from layers.layer import Layer

class Padding(Enum):
    VALID = "valid", lambda a, b: (a[0] - b[0] + 1, a[1] - b[1] + 1)
    SAME  = "same",  lambda a, _: (a)
    FULL  = "full",  lambda a, b: (a[0] + b[0] - 1, a[1] + b[1] - 1)

    @property
    def mode(self):
        return self.value[0]
    
    @property
    def calculate_output_size(self):
        return self.value[1]

# Convolutional Relu
class CR(Layer):
    def __init__(self, qty__filters: int, filter_size: int, optimization_method: OptimizationMethod, input_shape:Tuple[int, int, int], padding: Padding = Padding.VALID):

        self.qty_filters = qty__filters
        self.filter_size = filter_size
        self.optimization_method = optimization_method
        
        # cache for back_prop
        self.last_input = None

        self.input_shape = input_shape
        chanells, heigth, width = input_shape
        self.chanells = chanells
        # TODO: this is *SUPPOSING* VALID PADDING
        self.output_shape = (qty__filters, *padding.calculate_output_size((heigth, width), (filter_size, filter_size)))
        self.filters_shape = (qty__filters, chanells, filter_size, filter_size)
        self.filters = np.random.randn(*self.filters_shape)

        self.padding = padding

    def get_output_shape(self):
        return self.output_shape

    def _correlate2d(self, a:ndarray, b:ndarray, padding:Padding):
        aux = np.correlate(a.flatten(), b.flatten(), padding.mode)
        print(a.shape)
        print(b.shape)
        aux = aux.reshape(padding.calculate_output_size(a.shape, b.shape))
        return aux
    
    def _convolve2d(self, a:ndarray, b:ndarray, padding:Padding):
        aux = np.convolve(a.flatten(), b.flatten(), padding.mode)
        aux = aux.reshape(padding.calculate_output_size(a.shape, b.shape))
        return aux

    def forward_prop(self, input: ndarray):
        # aplies the qty_filters filters to the input image
        self.last_input = input

        output = np.zeros(self.output_shape)
        for i in range(self.qty_filters):
            for j in range(self.chanells):
                print(input[j], self.filters[i,j], self.padding)
                output[i] += self._correlate2d(input[j], self.filters[i,j], self.padding)
        return output

    def back_prop(self, loss_gradient: ndarray):
        filters_gradient = np.zeros(self.filters_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.qty_filters):
            for j in range(self.chanells):
                filters_gradient[i, j] = self._correlate2d(self.last_input[j], loss_gradient[i], self.padding)
                input_gradient[j] += self._convolve2d(loss_gradient[i], self.filters[i, j], Padding.FULL)

        self.filters = self.optimization_method.get_updated_weights(self.filters, filters_gradient)
        return input_gradient