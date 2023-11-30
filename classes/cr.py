import numpy as np
from enum import Enum

class Padding(Enum):
    VALID = 0
    SAME  = 1
    FULL  = 2

# Convolutional Relu
class CR:
    def __init__(self, qty__filters: int, filter_size: int, padding: Padding = Padding.VALID):
        self.qty_filters = qty__filters
        self.filter_size = filter_size

        # array of qty_filters matrices of filter_size * filter_size
        # Divided by filter_size ^ 2 to reduce variance
        self.filters = np.random.randn(qty__filters, filter_size, filter_size) / (filter_size**2)

        self.padding = padding

    def iterate_image_regions(self, input_image):
        # generates matrices of filter_size * filter_size
        # for each section of the image, so that filters are aplied

        heigth, width = input_image.shape
        match(self.padding):
            case(Padding.VALID):
                heigth -= self.filter_size // 2 # floored divition
                width  -= self.filter_size // 2 # floored divition
            case(Padding.SAME):
                raise "Unimplemented"
            case(Padding.FULL):
                raise "Unimplemented"
            case(_):
                raise "Unimplemented"
            

        for i in range(heigth):
            for j in range(width):
                image_region = input_image[i:(i+self.filter_size), j:(j+self.filter_size)]
                yield image_region, i, j

    def foward_prop():
        pass