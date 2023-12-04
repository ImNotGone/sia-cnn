from abc import ABC
from abc import abstractmethod

from numpy import ndarray
import numpy as np


class OptimizationMethod(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_updated_weights(self, weights: ndarray, gradient_weights: ndarray, loss: float=0) -> ndarray:
        pass


# ----- Gradient Descent -----
class GradientDescent(OptimizationMethod):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def get_updated_weights(self, weights: ndarray, gradient_weights: ndarray, loss: float=0) -> ndarray:
        return weights - self.learning_rate * gradient_weights


# ----- Momentum -----
class Momentum(OptimizationMethod):
    def __init__(self, learning_rate: float, momentum: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.previous_gradient = np.array([])

    def get_updated_weights(self, weights: ndarray, gradient_weights: ndarray, loss: float=0) -> ndarray:
        if self.previous_gradient.size == 0:
            self.previous_gradient = np.zeros_like(weights)

        updated_weights = weights - (self.learning_rate * gradient_weights + self.momentum * self.previous_gradient)
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

    def get_updated_weights(self, weights: ndarray, gradient_weights: ndarray, loss: float=0) -> ndarray:
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


# ----- Adaptive Eta -----
class AdaptiveEta(OptimizationMethod):
    def __init__(self, initial_eta: float, decay_factor: float = 0.9, increase_factor: float = 1.1, threshold: float = 0.01):
        super().__init__()
        self.learning_rate = initial_eta
        self.decay_factor = decay_factor
        self.increase_factor = increase_factor
        self.threshold = threshold
        self.previous_loss = float('inf')

    def get_updated_weights(self, weights: ndarray, gradient_weights: ndarray, loss: float=0) -> ndarray:

        current_loss = loss
        loss_change = self.previous_loss - current_loss

         # Check if loss is increasing and, if it is, decrease learning rate
        if loss_change > self.threshold:
            self.learning_rate *= self.decay_factor

        # Check if loss is decreasing and, if it is, increase learning rate
        elif loss_change < -self.threshold:
            self.learning_rate *= self.increase_factor

        self.previous_loss = current_loss 

        updated_weights = weights - self.learning_rate * gradient_weights
        return updated_weights




        
        
