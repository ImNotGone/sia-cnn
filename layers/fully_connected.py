from numpy import ndarray
import numpy as np

class FullyConnected():

    # TODO: configurar la funcion de activacion
    def sigmoid(self, input: ndarray):
        return 1 / (1 + np.exp(-input))

    def sigmoid_prime(self, input: ndarray):
        return self.sigmoid(input) * (1 - self.sigmoid(input))

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

        # Para guardar los valores de entrada y salida
        self.input = np.zeros(input_size)
        self.excitements = np.zeros(output_size)
        self.output = np.zeros(output_size)

        self.weights = np.random.randn(input_size, output_size)

    def forward_prop(self, input: ndarray):
        self.input = input

        excitements = input @ self.weights
        activations = self.sigmoid(excitements)

        self.output = activations
        self.excitements = excitements

        return activations

    def back_prop(self, loss_gradient:ndarray, learning_rate: float):

        # Loss en funcion de los excitements
        gradient_excitements = loss_gradient * self.sigmoid_prime(self.excitements)

        # Loss en funcion de los pesos
        # Reshape para que sea una matriz de dim (input_size, output_size)
        gradient_weights = self.input.reshape(-1, 1) @ gradient_excitements.reshape(1, -1)

        # Por ahora gradient descent
        self.weights -= learning_rate * gradient_weights

        # Loss en funcion de la entrada,
        gradient_input = gradient_excitements @ self.weights.T

        return gradient_input



