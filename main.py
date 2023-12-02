import json

from cnn import CNN
from dataset_loader import load_dataset
from layers.cr import CR
from layers.pl import PL
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from layers.softmax import SM
import numpy as np
from layers.utils.activation_functions import get_act_func
from layers.utils.optimization_methods import Adam, GradientDescent, Momentum
from plots import visualize_first_layer_filters, visualize_feature_maps


def get_batch_size(config, dataset_size) -> int:
    training_strategy = config["training_strategy"]

    if training_strategy == "batch":
        return dataset_size
    elif training_strategy == "mini_batch":
        batch_size = config["batch_size"]

        if batch_size > dataset_size:
            raise Exception("Batch size is bigger than dataset size")

        return batch_size
    elif training_strategy == "online":
        return 1
    else:
        raise Exception("Training strategy not found")

def main():
    training_data, training_labels, test_data, test_labels = load_dataset()

    data_shape = training_data.shape[1:]

    config_file = "config.json"

    with open(config_file) as json_file:
        config = json.load(json_file)

        batch_size = get_batch_size(config, len(training_data))
        epochs = config["epochs"]
        delta = config["delta"]
        activation_function = get_act_func(config)

        cnn = CNN(
            [
                CR(5, 3, Adam(delta), (1, 50, 50)),
                PL((5, 48, 48)),
                CR(3, 3, Adam(delta), (5, 24, 24)),
                PL((3, 22, 22)),
                Flatten(),
                FullyConnected(
                    363,
                    100,
                    activation_function,
                    Adam(delta),
                ),
                FullyConnected(100, 50, activation_function, Adam(delta)),
                FullyConnected(50, 5, activation_function, Adam(delta)),
                FullyConnected(5, 1, activation_function, Adam(delta)),
            ]
        )

    loss_per_epoch = cnn.train(training_data, training_labels, epochs, batch_size)

    total_predictions = 0
    for data, label in zip(test_data, test_labels):
        data = np.array([data])
        output = cnn.forward_prop(data)

        predicted = "square" if output < 0.5 else "triangle"
        actual = "square" if label == 0 else "triangle"

        print(f"Predicted: {predicted}, Output: {output}")
        print(f"Actual: {actual}, Label: {label}")

        if predicted == actual:
            total_predictions += 1
        print()

    print("Accuracy: ", total_predictions / len(test_data))

    visualize_first_layer_filters(cnn)

    # Get a square and a triangle
    square = test_data[0]
    for data, label in zip(test_data, test_labels):
        if label == 0:
            square = data
            break
    triangle = test_data[0]
    for data, label in zip(test_data, test_labels):
        if label == 1:
            triangle = data
            break

    visualize_feature_maps(cnn, square, "square")
    visualize_feature_maps(cnn, triangle, "triangle")

    print("Training finished")


if __name__ == "__main__":
    main()
