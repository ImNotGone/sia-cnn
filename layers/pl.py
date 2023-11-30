from enum import Enum
import numpy as np
from numpy import ndarray

from layers.utils import ForwardPropNotDoneError
from layers.layer import Layer


class PoolingType(Enum):
    MAX = np.amax
    MIN = np.amin
    AVG = np.average


class PL(Layer):
    def __init__(self, pooling_type: PoolingType = PoolingType.MAX, stride: int = 2):
        self.stride = stride
        self.pooling_func = pooling_type.value

        # cache for back_prop
        self.last_input_image = None

    def iterate_image_regions(self, input_image: ndarray):

        heigth, width, _ = input_image.shape
        # pooling reduces size by the stride factor
        heigth = heigth // self.stride  # floored division
        width = width // self.stride  # floored division

        for i in range(heigth):
            for j in range(width):
                i_start = i * self.stride
                i_end = i * self.stride + self.stride
                j_start = j * self.stride
                j_end = j * self.stride + self.stride
                image_region = input_image[i_start:i_end, j_start:j_end]
                yield image_region, i, j

    def forward_prop(self, input_image: ndarray):

        # cache'd for easier back_prop
        self.last_input_image = input_image

        heigth, width, qty_filters = input_image.shape
        # pooling reduces size by the stride factor
        heigth = heigth // self.stride  # floored division
        width = width // self.stride  # floored division

        output = np.zeros((heigth, width, qty_filters))

        for image_region, i, j in self.iterate_image_regions(input_image):
            output[i, j] = self.pooling_func(image_region, axis=(0, 1))  # apply function on axis 0 & 1

        return output

    def back_prop(self, loss_gradient: ndarray):
        # We cached the last input image while doing forward_prop to make back_prop easier
        # We check that forward propagation was done before doing back propagation
        if (self.last_input_image is None):
            # TODO implement error
            raise ForwardPropNotDoneError

        # We create the previous filters layer to reconstruct it with the same shape as the current filters
        input = np.zeros(self.last_input_image.shape)

        # Now we reconstruct the filters, using the cached last input image
        for image_region, i, j in self.iterate_image_regions(self.last_input_image):
            height, width, k = image_region.shape
            pool_value = self.pooling_func(image_region, axis=(0, 1))

            for i_2 in range(height):
                for j_2 in range(width):
                    for k_2 in range(k):
                        # If pixel is MAX, MIN or AVG, copy gradient to it.
                        if image_region[i_2, j_2, k_2] == pool_value[k_2]:
                            input[i * 2 + i_2, j * 2 + j_2, k_2] = loss_gradient[i, j, k_2]

        self.last_input_image = None

        return input
