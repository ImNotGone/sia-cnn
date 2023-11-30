from enum import Enum
import numpy as np
from numpy import ndarray

class PoolingType(Enum):
    MAX = 0
    MIN = 1
    AVG = 2

class PL():
    def __init__(self, pooling_type: PoolingType = PoolingType.MAX, stride: int = 2):
        self.stride = stride

        self.pooling_func = None

        match(pooling_type):
            case(PoolingType.MAX):
                self.pooling_func = np.amax
            case(PoolingType.MIN):
                self.pooling_func = np.amin
            case(PoolingType.AVG):
                self.pooling_func = np.average
            case(_):
                raise "Unimplemented"

    def iterate_image_regions(self, input_image:ndarray):

        heigth, width, _ = input_image.shape
        # pooling reduces size by the stride factor
        heigth = heigth // self.stride # floored division
        width  = width  // self.stride # floored division

        for i in range(heigth):
            for j in range(width):
                i_start = i*self.stride
                i_end   = i*self.stride + self.stride
                j_start = j*self.stride
                j_end   = j*self.stride + self.stride
                image_region = input_image[i_start:i_end, j_start:j_end]
                yield image_region, i, j

    def foward_prop(self, input_image:ndarray):

        # cache'd for easier back_prop
        self.last_input_image = input_image

        heigth, width, qty_filters = input_image.shape
        # pooling reduces size by the stride factor
        heigth = heigth // self.stride # floored division
        width  = width  // self.stride # floored division

        output = np.zeros((heigth, width, qty_filters))

        for image_region, i, j in self.iterate_image_regions(input_image):
            output[i, j] = self.pooling_func(image_region, axis=(0, 1)) # apply function on axis 0 & 1

        return output

    def back_prop(self):
        pass