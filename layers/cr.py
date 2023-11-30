import numpy as np
from enum import Enum
from numpy import ndarray

from layers.utils import ForwardPropNotDoneError
from layers.utils.optimization_methods import OptimizationMethod
from layers.layer import Layer

class Padding(Enum):
    VALID = 0
    SAME  = 1
    FULL  = 2

# Convolutional Relu
class CR(Layer):
    def __init__(self, qty__filters: int, filter_size: int, optimization_method: OptimizationMethod, padding: Padding = Padding.VALID):
        self.qty_filters = qty__filters
        self.filter_size = filter_size

        self.optimization_method = optimization_method
        
        # cache for back_prop
        self.last_input_image = None

        # array of qty_filters matrices of filter_size * filter_size
        # Divided by filter_size ^ 2 to reduce variance
        self.filters = np.random.randn(qty__filters, filter_size, filter_size) / (filter_size**2)

        self.padding = padding

    def iterate_image_regions(self, input_image: ndarray):
        # generates matrices of filter_size * filter_size
        # for each section of the image, so that filters are aplied

        heigth, width = input_image.shape
        match(self.padding):
            case(Padding.VALID):
                # reduce size, for borders to fit
                heigth -= (self.filter_size // 2)*2 # floored division
                width  -= (self.filter_size // 2)*2 # floored division
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

    def forward_prop(self, input_image: ndarray):
        # aplies the qty_filters filters to the input image

        heigth, width = input_image.shape
        match(self.padding):
            case(Padding.VALID):
                # reduce size, for borders to fit
                heigth -= (self.filter_size // 2)*2 # floored division
                width  -= (self.filter_size // 2)*2 # floored division
                print(heigth, width, self.filter_size, (self.filter_size // 2)*2)
                print(self.filters)
            case(_):
                raise 'Unimplemented'

        # cache'd for easier back_prop
        self.last_input_image = input_image

        output = np.zeros((heigth, width, self.qty_filters))

        for image_region, i, j in self.iterate_image_regions(input_image):
            # np.sum in this case
            # compresses a list of matrices to a list of the sum of each matrix
            output[i, j] = np.sum(image_region * self.filters, axis=(1, 2)) # sum along axis 1 & 2
    
        return output

    def back_prop(self, loss_gradient: ndarray):
        #We cached the last input image while doing forward_prop to make back_prop easier
        #We check that forward propagation was done before doing back propagation
        if(self.last_input_image is None):
            #TODO implement error
            raise ForwardPropNotDoneError
        

        #We create the previous filters layer to reconstruct it with the same shape as the current filters
        previous_filters = np.zeros(self.filters.shape)


        #Now we reconstruct the filters, using the cached last input image
        for image_region, i, j in self.iterate_image_regions(self.last_input_image):
            for k in range(self.qty_filters):
                previous_filters[k] +=  loss_gradient[i, j, k] * image_region
        
        # floatNow we update the current filters that we are on (going back to the previous set)
        self.filters = self.optimization_method.get_updated_weights(self.filters, previous_filters)

        
        #TODO check reset last input image
        self.last_input_image=None

        return previous_filters
