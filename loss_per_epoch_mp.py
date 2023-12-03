from cnn import CNN
from dataset_loader import load_dataset
from layers.convolutional import Convolutional
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from layers.pooling import Pooling
from layers.softmax import SM
from layers.utils.activation_functions import ReLU, Sigmoid
from copy import deepcopy
from layers.utils.optimization_methods import Adam, GradientDescent, Momentum
""" from plots import plot_errors_per_architecture """


import numpy as np
import json
import multiprocessing

def architecture_test():
    
    delta = 0.001
    activation_function=Sigmoid()
    
    iterations = 10
    architectures = [
        ([CNN(
        [
            Convolutional(5, 3, Adam(delta), (1, 50, 50)),
            Pooling((5, 48, 48)),
            Convolutional(3, 3, Adam(delta), (5, 24, 24)),
            Pooling((3, 22, 22)),
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
        ) for _ in range(iterations)], "cnn1"),
        ([CNN(
        [
            Convolutional(5, 3, Adam(delta), (1, 50, 50)),
            Pooling((5, 48, 48)),
            Convolutional(3, 3, Adam(delta), (5, 24, 24)),
            Pooling((3, 22, 22)),
            Flatten(),
            FullyConnected(
                363,
                50,
                activation_function,
                Adam(delta),
            ),
            FullyConnected(50, 10, activation_function, Adam(delta)),
            FullyConnected(10, 1, activation_function, Adam(delta)),

        ]
        ) for _ in range(iterations)], "cnn2"),
    ]

    training_data, training_labels, test_data, test_labels = load_dataset()

    data_shape = training_data.shape[1:]

    epochs = 3


    batch_size = training_data.shape[0]
    

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
                        batch_size,
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

    """ plot_errors_per_architecture(mean_errors_per_architecture) """

    # Serialize errors
    with open("errors_architecture.json", "w") as f:
        json.dump(mean_errors_per_architecture, f)



def train_and_calculate_error(
    cnn,
    training_data,
    training_labels,
    epochs,
    batch_size,
    errors_list,
):
    # Get a random seed for the process
    pid = multiprocessing.current_process().pid
    np.random.seed(pid)

    loss_per_epoch = cnn.train(training_data, training_labels, epochs, batch_size)

    best_error = min(loss_per_epoch)

    errors_list.append(best_error)

# No funciona si usamos lambdas en pooling

if __name__ == "__main__":
    architecture_test()
