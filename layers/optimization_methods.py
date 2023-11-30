from abc import ABC
from abc import abstractmethod

from numpy import ndarray
import numpy as np


class OptimizationMethod(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_updated_weights(self, weights: ndarray, gradient_weights: ndarray) -> ndarray:
        pass

# ----- Gradient Descent -----
class GradientDescent(OptimizationMethod):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def get_updated_weights(self, weights: ndarray, gradient_weights: ndarray) -> ndarray:
        return weights - self.learning_rate * gradient_weights

# ----- Momentum -----
class Momentum(OptimizationMethod):
    def __init__(self, learning_rate: float, momentum: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.previous_gradient = np.array([])

    def get_updated_weights(self, weights: ndarray, gradient_weights: ndarray) -> ndarray:
        updated_weights = weights - self.learning_rate * gradient_weights + self.momentum * self.previous_gradient
        self.previous_gradient = gradient_weights
        return updated_weights

# ----- Adam -----
class Adam(OptimizationMethod):
    def __init__(self, learning_rate: float, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = np.array([])
        self.v = np.array([])
        self.t = 0

    def get_updated_weights(self, weights: ndarray, gradient_weights: ndarray) -> ndarray:
        self.t += 1

        if self.m.size == 0:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient_weights
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(gradient_weights)
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        updated_weights = weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated_weights



        
        
