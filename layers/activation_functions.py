
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray


class ActivationFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def call(self, x: ndarray):
        pass

    @abstractmethod
    def derivative(self, x: ndarray):
        pass


# ------ ReLU ------

class ReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def call(self, x: ndarray):
        return x * (x > 0)

    def derivative(self, x: ndarray):
        return 1. * (x > 0)

# ----- Sigmoid -----
class Sigmoid(ActivationFunction):
    def __init__(self, beta: float = 1):
        super().__init__()
        self.beta = beta

    def call(self, x: ndarray):
        return 1 / (1 + np.exp(-self.beta * x))

    def derivative(self, x: ndarray):
        return self.beta * self.call(x) * (1 - self.call(x))

# ----- Tanh -----
class Tanh(ActivationFunction):
    def __init__(self, beta: float = 1):
        super().__init__()
        self.beta = beta

    def call(self, x: ndarray):
        return np.tanh(self.beta * x)

    def derivative(self, x: ndarray):
        return self.beta * (1 - self.call(x) ** 2)
