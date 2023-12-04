from typing import List, Tuple

import numpy as np
import copy
from numpy import ndarray

from layers.layer import Layer
from layers.convolutional import Convolutional


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

    def __init__(self, layers: List[Layer], input_shape: Tuple):
        # Aca van las layers, pueden ser convolucionales, pooling, fully connected, softmax, etc
        self.layers = layers

        # Inicializo las layers
        for layer in self.layers:
            layer.initialize(input_shape)
            input_shape = layer.get_output_shape()

    def forward_prop(self, input: ndarray):
        current_output = input

        for layer in self.layers:
            current_output = layer.forward_prop(current_output)

        return current_output

    def back_prop(self, loss: ndarray):
        current_gradient = loss

        for layer in reversed(self.layers):
            current_gradient = layer.back_prop(current_gradient)

    def iterate_mini_batches(self, data, labels, batch_size):
        assert data.shape[0] == labels.shape[0]

        if batch_size == len(data):
            yield data, labels
            return

        for start_idx in range(0, data.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, data.shape[0])
            excerpt = slice(start_idx, end_idx)
            yield data[excerpt], labels[excerpt]

    def train(self, data: ndarray, labels: ndarray, epochs: int, batch_size: int):
        loss_per_epoch = []
        best_loss = np.inf
        best_model = None

        for epoch in range(epochs):
            losses = []

            print(f"Starting epoch {epoch + 1}")

            for batch_data, batch_labels in self.iterate_mini_batches(
                data, labels, batch_size
            ):
                for i, sample, label in zip(
                    range(len(batch_data)), batch_data, batch_labels
                ):
                    sample = np.array([sample])
                    output = self.forward_prop(sample)

                    loss = self.cross_entropy_loss(output, label)
                    losses.append(loss)

                    print(f"{i + 1}/{len(batch_data)}", end="\r")

                    # Para backprop
                    loss_gradient = self.cross_entropy_loss_gradient(output, label)
                    self.back_prop(loss_gradient)

            loss = np.mean(losses)
            loss_per_epoch.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_model = copy.deepcopy(self)

            #Adaptive learning rate (ETA)
            #TODO: should the learning rate be updated every iteration or after consistent loss increase/decrease?
            # Check if loss is increasing and, if it is, decrease learning rate
            if epoch > 0 and loss_per_epoch[-1] > loss_per_epoch[-2]:
                for layer in self.layers:
                    if hasattr(layer, 'optimization_method'):
                        layer.optimization_method.update_learning_rate(layer.optimization_method.learning_rate * 0.9)
                        print(f"New learning rate for layer {layer.__class__.__name__}: {layer.optimization_method.learning_rate}")

            # Check if loss is decreasing and, if it is, increase learning rate
            if epoch > 0 and loss_per_epoch[-1] < loss_per_epoch[-2]:
                for layer in self.layers:
                    if hasattr(layer, 'optimization_method'):
                        layer.optimization_method.update_learning_rate(layer.optimization_method.learning_rate * 1.1)
                        print(f"New learning rate for layer {layer.__class__.__name__}: {layer.optimization_method.learning_rate}")

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
            if isinstance(layer, Convolutional):
                # Apply ReLU
                current_output = np.maximum(current_output, 0)
                layer_feature_maps = [
                    current_output[i] for i in range(current_output.shape[0])
                ]
                feature_maps.append(layer_feature_maps)

        return feature_maps

    def get_filters(self):
        filters = []

        for layer in self.layers:
            if isinstance(layer, Convolutional):
                filters.append(layer.filters)

        return filters
