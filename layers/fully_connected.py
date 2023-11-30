from numpy import ndarray
import numpy as np

from layers.utils.activation_functions import ActivationFunction
from layers.utils.optimization_methods import OptimizationMethod
from layers.layer import Layer


class FullyConnected(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_function: ActivationFunction,
        optimization_method: OptimizationMethod,
    ):
        self.input_size = input_size
        self.output_size = output_size

        self.activation_function = activation_function
        self.optimization_method = optimization_method

        # Para guardar los valores de entrada y salida
        self.input = np.zeros(input_size)
        self.excitements = np.zeros(output_size)
        self.output = np.zeros(output_size)

        self.weights = np.random.randn(input_size, output_size)

    def forward_prop(self, input: ndarray):
        self.input = input

        excitements = input @ self.weights
        activations = self.activation_function.call(excitements)

        self.output = activations
        self.excitements = excitements

        return activations

    def back_prop(self, loss_gradient: ndarray):
        # Loss en funcion de los excitements
        gradient_excitements = loss_gradient * self.activation_function.derivative(
            self.excitements
        )

        # Loss en funcion de los pesos
        # Reshape para que sea una matriz de dim (input_size, output_size)
        gradient_weights = self.input.reshape(-1, 1) @ gradient_excitements.reshape(
            1, -1
        )

        # Actualizar pesos
        self.weights = self.optimization_method.get_updated_weights(
            self.weights, gradient_weights
        )

        # Loss en funcion de la entrada,
        gradient_input = gradient_excitements @ self.weights.T

        return gradient_input
