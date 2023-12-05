from cnn import CNN
from utils.dataset_loader import load_dataset
from layers.convolutional import Convolutional
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from layers.pooling import Pooling
from layers.softmax import SM
from layers.relu import Relu
from layers.utils.activation_functions import ReLU, Sigmoid
from copy import deepcopy
from layers.utils.optimization_methods import (
    Adam,
    GradientDescent,
    Momentum,
    AdaptiveEta,
)
from utils.plots import plot_errors_per_architecture
from utils.save import save_errors_per_architecture


import numpy as np
import multiprocessing


def architecture_test():
    training_data, training_labels, test_data, test_labels = load_dataset()
    data_shape = np.array([training_data[0]]).shape
    print(data_shape)

    delta = 0.001
    activation_function = Sigmoid()

    iterations = 10
    epochs = 10
    batch_size = training_data.shape[0]

    architectures = [
        (
            [
                CNN(
                    [
                        Pooling(),
                        Pooling(),
                        Convolutional(5, 3, Adam(delta)),
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
                for _ in range(iterations)
            ],
            "2MP-5C3-R-2MP-FL-100FC-1FC",
        ),
        (
            [
                CNN(
                    [
                        Pooling(),
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
                for _ in range(iterations)
            ],
            "2MP-3C3-R-2MP-FL-100FC-1FC",
        ),
    ]
    data_shape = training_data.shape[1:]

    mean_errors_per_architecture = {}

    for architecture_set, name in architectures:
        errors = multiprocessing.Manager().list()

        # Create processes
        processes = []
        for cnn in architecture_set:
            processes.append(
                multiprocessing.Process(
                    target=train_and_calculate_error,
                    args=(
                        cnn,
                        training_data,
                        training_labels,
                        epochs,
                        errors,
                    ),
                )
            )

        # Start threads
        for process in processes:
            process.start()

        # Wait for threads to finish
        for process in processes:
            process.join()

        mean_error = np.mean(errors)
        std_error = np.std(errors)

        mean_errors_per_architecture[name] = (mean_error, std_error)

    # Serialize errors
    save_errors_per_architecture(mean_errors_per_architecture)

    plot_errors_per_architecture(mean_errors_per_architecture)


def train_and_calculate_error(
    cnn,
    training_data,
    training_labels,
    epochs,
    errors_list,
):
    # Get a random seed for the process
    pid = multiprocessing.current_process().pid
    np.random.seed(pid)

    loss_per_epoch = cnn.train(training_data, training_labels, epochs)

    best_error = min(loss_per_epoch)

    errors_list.append(best_error)


# No funciona si usamos lambdas en pooling

if __name__ == "__main__":
    architecture_test()
