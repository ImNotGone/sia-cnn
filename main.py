import json
import time

from cnn import CNN
from utils.dataset_loader import load_dataset
from layers.convolutional import Convolutional
from layers.pooling import Pooling
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from layers.softmax import SM
from layers.relu import Relu
import numpy as np
from layers.utils.activation_functions import get_act_func
from layers.utils.optimization_methods import Adam, GradientDescent, Momentum
from utils.plots import (
    plot_confusion_matrix,
    visualize_first_layer_filters,
    visualize_feature_maps,
    plot_errors_per_epoch,
)
from utils.save import save_errors_per_epoch, save_predictions


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
    start_time = time.time()
    config_file = "config.json"

    with open(config_file) as json_file:
        config = json.load(json_file)

        print("Loading dataset")

        img_size = config["image_size"]

        training_data, training_labels, test_data, test_labels = load_dataset(img_size)

        data_shape = np.array([training_data[0]]).shape

        batch_size = get_batch_size(config, len(training_data))
        epochs = config["epochs"]
        delta = config["delta"]
        activation_function = get_act_func(config)

        cnn = CNN(
            [
                Convolutional(5, 3, Adam(delta)),
                Relu(),
                Pooling(),
                Convolutional(3, 3, Adam(delta)),
                Relu(),
                Pooling(),
                Flatten(),
                FullyConnected(100, activation_function, Adam(delta)),
                FullyConnected(
                    1,
                    activation_function,
                    Adam(delta),
                ),
            ],
            data_shape,
        )

    print("Starting training")

    loss_per_epoch = cnn.train(training_data, training_labels, epochs, batch_size)

    plot_errors_per_epoch(loss_per_epoch)
    save_errors_per_epoch(loss_per_epoch)

    # --- Test ---

    print("Starting test")

    predictions = []
    correct = 0
    for data, label in zip(test_data, test_labels):
        data = np.array([data])
        output = cnn.forward_prop(data)

        predicted = "square" if output < 0.5 else "triangle"
        actual = "square" if label == 0 else "triangle"

        if predicted == actual:
            correct += 1

        predictions.append((predicted, actual, output, label))

    plot_confusion_matrix(predictions)
    save_predictions(predictions)

    print(f"Accuracy: {correct / len(test_data)}")

    # --- Visualization ---

    print("Starting visualization")

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

    end_time = time.time()
    print(f"Finished in {int(end_time - start_time)} seconds")


if __name__ == "__main__":
    main()
