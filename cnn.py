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

    def __init__(self, layers: List[Layer]):

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

    def train(self, data: ndarray, labels: ndarray, epochs: int, batch_size: int):
        loss_per_epoch = []
        best_loss = np.inf
        best_model = None

        for epoch in range(epochs):
            losses = []

            print(f"Starting epoch {epoch + 1}")

            for i, sample, label in zip(range(len(data)), data, labels):
                sample = np.array([sample])
                output = self.forward_prop(sample)

                loss = self.cross_entropy_loss(output, label)
                losses.append(loss)

                print(f"{i + 1}/{len(data)}", end="\r")

                # Para backprop
                loss_gradient = self.cross_entropy_loss_gradient(output, label)
                self.back_prop(loss_gradient)

            loss = np.mean(losses)
            loss_per_epoch.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_model = copy.deepcopy(self)

            print(f"Finished Epoch: {epoch + 1}, Avg Loss: {loss_per_epoch[-1]}")

        if best_model is not None:
            self = best_model

        return loss_per_epoch

    def get_feature_maps(self, input: ndarray):

        input = np.array([input])
        feature_maps = []

        current_output = input
        for layer in self.layers:
            current_output = layer.forward_prop(current_output)
            if isinstance(layer, CR):
                layer_feature_maps = [current_output[i] for i in range(current_output.shape[0])]
                feature_maps.append(layer_feature_maps)

        return feature_maps

    def get_filters(self):
        filters = []

        for layer in self.layers:
            if isinstance(layer, CR):
                filters.append(layer.filters)

        return filters

        

