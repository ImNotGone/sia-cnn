from numpy import ndarray
import numpy as np

from layers.layer import Layer
from layers.utils.optimization_methods import OptimizationMethod


class SM(Layer):
    def softmax(self, input: ndarray):
        return np.exp(input) / np.sum(np.exp(input))

    def __init__(
        self, input_size: int, output_size: int, optimization_method: OptimizationMethod
    ):
        self.input_size = input_size
        self.output_size = output_size

        self.optimization_method = optimization_method

        # Para guardar los valores de entrada y salida
        self.input = np.zeros(input_size)
        self.output = np.zeros(output_size)

        self.weights = np.random.randn(input_size, output_size)

    def forward_prop(self, input: ndarray):
        self.input = input

        excitements = input @ self.weights
        activations = self.softmax(excitements)

        self.output = activations

        return activations

    def back_prop(self, loss_gradient: ndarray):
        # Loss en funcion de los pesos
        """ gradient_weights = self.input.T @ loss_gradient """
        gradient_weights = self.input.reshape(-1, 1) @ loss_gradient.reshape(
            1, -1
        )

        # Actualizar pesos
        self.weights = self.optimization_method.get_updated_weights(
            self.weights, gradient_weights
        )

        # Loss en funcion de la entrada,
        # con softmax no hay que multiplicar por la derivada (se simplifica)
        gradient_input = loss_gradient @ self.weights.T

        return gradient_input
