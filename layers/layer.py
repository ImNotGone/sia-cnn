# Abstract class layer
from abc import ABC, abstractmethod
from numpy import ndarray

class Layer(ABC)

    @abstractmethod
    def forward_prop(self, input: ndarray):
        raise NotImplementedError

    @abstractmethod
    def back_prop(self, loss_gradient: ndarray, learning_rate: float):
        raise NotImplementedError
