import numpy as np
from enum import Enum
from typing import Tuple
from numpy import ndarray

from layers.utils import ForwardPropNotDoneError
from layers.utils.optimization_methods import OptimizationMethod
from layers.layer import Layer

class Padding(Enum):
    VALID = "valid", lambda a, b: (a[0] - b[0] + 1, a[1] - b[1] + 1), lambda _: (0, 0)
    SAME  = "same",  lambda a, _: (a),                                lambda b: ((b[0] - 1) // 2, (b[1] - 1) // 2)
    FULL  = "full",  lambda a, b: (a[0] + b[0] - 1, a[1] + b[1] - 1), lambda b: (b[0] - 1, b[1] - 1)

    @property
    def mode(self):
        return self.value[0]
    
    @property
    def calculate_output_size(self):
        return self.value[1]

    @property
    def calculate_padding(self):
        return self.value[2]

class Convolutional(Layer):
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

    # Fast implementation of 2d correlation using numpy
    # Source: https://medium.com/@thepyprogrammer/2d-image-convolution-with-numpy-with-a-handmade-sliding-window-view-946c4acb98b4
    def _correlate2d(self, input:ndarray, filter:ndarray, padding:Padding):
        output_heigth, output_width = padding.calculate_output_size(input.shape, filter.shape)
        vertical_padding, horizontal_padding = padding.calculate_padding(filter.shape)
        filter_heigth, filter_width = filter.shape

        input = np.pad(input, ((vertical_padding, vertical_padding), (horizontal_padding, horizontal_padding)), mode='constant', constant_values=0)

        # filter1 creates an index system that calculates the sum of the x and y indices at each point
        # Shape of filter1 is h x kernelW
        filter1 = np.arange(filter_width) + np.arange(output_heigth)[:, np.newaxis]
          
        # filter2 similarly creates an index system
        # Shape of filter2 is w * kernelH
        filter2 = np.arange(filter_heigth) + np.arange(output_width)[:, np.newaxis]

        # intermediate is the stepped data, which has the shape h x kernelW x imageH
        intermediate = input[filter1]
          
        # transpose the inner dimensions of the intermediate so as to enact another filter
        # shape is now h x imageH x kernelW
        intermediate = np.transpose(intermediate, (0, 2, 1))
          
        # Apply filter2 on the inner data piecewise, resultant shape is h x w x kernelW x kernelH
        intermediate = intermediate[:, filter2]
          
        # transpose inwards again to get a resultant shape of h x w x kernelH x kernelW
        intermediate = np.transpose(intermediate, (0, 1, 3, 2))
          
        # piecewise multiplication with kernel
        product = intermediate * filter
          
        # find the sum of each piecewise product, shape is now h x w
        output = product.sum(axis = (2,3))

        return output
    
    def _convolve2d(self, a:ndarray, b:ndarray, padding:Padding): 
        return self._correlate2d(a, b.T, padding)

    def forward_prop(self, input: ndarray):
        if (input.shape != self.input_shape):
            print(f"actual_input_shape: {input.shape}")
            print(f"expected_input_shape: {self.input_shape}")
            raise "input shape specified does not match"
        # aplies the qty_filters filters to the input image
        self.last_input = input

        output = np.zeros(self.output_shape)
        for i in range(self.qty_filters):
            for j in range(self.chanells):
                output[i] += self._correlate2d(input[j], self.filters[i,j], self.padding)
        return output

    def back_prop(self, loss_gradient: ndarray):
        if (self.last_input is None):
            raise ForwardPropNotDoneError
        filters_gradient = np.zeros(self.filters_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.qty_filters):
            for j in range(self.chanells):
                filters_gradient[i, j] = self._correlate2d(self.last_input[j], loss_gradient[i], self.padding)
                input_gradient[j] += self._convolve2d(loss_gradient[i], self.filters[i, j], Padding.FULL)

        self.filters = self.optimization_method.get_updated_weights(self.filters, filters_gradient)

        self.last_input = None
        return input_gradient
