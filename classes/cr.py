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

        # array of qty_filters matrices of filter_size * filter_size
        # Divided by filter_size ^ 2 to reduce variance
        self.filters = np.random.randn(qty__filters, filter_size, filter_size) / (filter_size**2)

        self.padding = padding

    def iterator_regions():
        pass

    def foward_prop():
        pass