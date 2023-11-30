from typing import List

import numpy as np
import copy
from numpy import ndarray

from layers.layer import Layer
from layers.cr import CR


class CNN:
    def cross_entropy_loss(self, predicted: ndarray, actual: ndarray):
        predicted = np.clip(predicted, 1e-15, 1 - 1e-15)  # Evito log(0)
        loss = -np.sum(
            actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)
        ) / len(predicted)
        return loss

    def cross_entropy_loss_gradient(self, predicted: ndarray, actual: ndarray):
        predicted = np.clip(predicted, 1e-15, 1 - 1e-15)  # Evito division por 0
        return (predicted - actual) / (predicted * (1 - predicted)) / len(predicted)

    def __init__(self, input_shape, num_classes: int, layers: List[Layer]):
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Aca van las layers, pueden ser convolucionales, pooling, fully connected, softmax, etc
        self.layers = layers

    def forward_prop(self, input: ndarray):
        current_output = input

        for layer in self.layers:
            current_output = layer.forward_prop(current_output)

        return current_output

    def back_prop(self, loss: ndarray):
        current_gradient = loss

        for layer in reversed(self.layers):
            current_gradient = layer.back_prop(current_gradient)

    def train(self, data: ndarray, labels: ndarray, epochs: int):
        loss_per_epoch = []
        best_loss = np.inf
        best_model = None

        for epoch in range(epochs):
            losses = []

            for sample, label in zip(data, labels):
                output = self.forward_prop(sample)

                loss = self.cross_entropy_loss(output, label)
                losses.append(loss)

                # Para backprop
                loss_gradient = self.cross_entropy_loss_gradient(output, label)
                self.back_prop(loss_gradient)

            loss = np.mean(losses)
            loss_per_epoch.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_model = copy.deepcopy(self)

            # Cada 5% imprimo el progreso
            if epoch % (epochs // 20) == 0:
                print(f"Epoch: {epoch} Loss: {loss_per_epoch[-1]}")

        if best_model is not None:
            self = best_model

        return loss_per_epoch

    def get_convolutions(self, input: ndarray):

        convolutions = []

        current_output = input
        for layer in self.layers:
            if isinstance(layer, CR):
                convolutions.append(current_output)
            current_output = layer.forward_prop(current_output)

        return convolutions

    def get_filters(self):
        filters = []

        for layer in self.layers:
            if isinstance(layer, CR):
                filters.append(layer.filters)

        return filters

        

