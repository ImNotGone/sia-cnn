from numpy import ndarray
import numpy as np

from layers.layer import Layer
from layers.utils.optimization_methods import OptimizationMethod


class SM(Layer):
    def softmax(self, input: ndarray):
        # Avoid overflow
        max_input = np.max(input)
        exp_input = np.exp(input - max_input)
        return exp_input / np.sum(exp_input)

    def __init__(self, output_size: int, optimization_method: OptimizationMethod):
        self.output_size = output_size
        self.optimization_method = optimization_method

    def initialize(self, input_size: int):
        self.input_size = input_size

        # Para guardar los valores de entrada y salida
        self.input = np.zeros(input_size)
        self.output = np.zeros(self.output_size)

        self.weights = np.random.randn(input_size, self.output_size)

    def get_output_shape(self):
        return self.output_size

    def forward_prop(self, input: ndarray):
        self.input = input

        excitements = input @ self.weights
        activations = self.softmax(excitements)

        self.output = activations

        return activations

    def back_prop(self, loss_gradient: ndarray):
        # Loss en funcion de los pesos
        """gradient_weights = self.input.T @ loss_gradient"""
        gradient_weights = self.input.reshape(-1, 1) @ loss_gradient.reshape(1, -1)

        # Actualizar pesos
        self.weights = self.optimization_method.get_updated_weights(
            self.weights, gradient_weights
        )

        # Loss en funcion de la entrada,
        # con softmax no hay que multiplicar por la derivada (se simplifica)
        gradient_input = loss_gradient @ self.weights.T

        return gradient_input
