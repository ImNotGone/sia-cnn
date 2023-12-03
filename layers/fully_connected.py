from numpy import ndarray
import numpy as np

from layers.utils.activation_functions import ActivationFunction
from layers.utils.optimization_methods import OptimizationMethod
from layers.layer import Layer


class FullyConnected(Layer):
    def __init__(
        self,
        neuron_qty: int,
        activation_function: ActivationFunction,
        optimization_method: OptimizationMethod,
    ):
        self.neuron_qty = neuron_qty

        self.activation_function = activation_function
        self.optimization_method = optimization_method

    def initialize(self, input_shape: int):
        self.input_shape = input_shape

        # Para guardar los valores de entrada y salida
        self.input = np.zeros(input_shape)
        self.excitements = np.zeros(self.neuron_qty)
        self.output = np.zeros(self.neuron_qty)

        self.weights = np.random.randn(input_shape, self.neuron_qty)

    def get_output_shape(self):
        return self.neuron_qty

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
