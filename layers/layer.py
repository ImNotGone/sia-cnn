# Abstract class layer
from abc import ABC, abstractmethod
from typing import Tuple
from numpy import ndarray

class Layer(ABC):

    @abstractmethod
    def forward_prop(self, input: ndarray):
        raise NotImplementedError

    @abstractmethod
    def back_prop(self, loss_gradient: ndarray):
        raise NotImplementedError

    @abstractmethod
    def initialize(self, input_shape):
        pass

    @abstractmethod
    def get_output_shape(self) -> Tuple:
        pass
