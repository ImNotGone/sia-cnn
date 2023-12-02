from cnn import CNN
from dataset_loader import load_dataset
from layers.cr import CR
from layers.pl import PL
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from layers.softmax import SM
import numpy as np
from layers.utils.activation_functions import ReLU, Sigmoid
from layers.utils.optimization_methods import Adam, GradientDescent, Momentum
from plots import visualize_first_layer_filters, visualize_feature_maps


def main():
    training_data, training_labels, test_data, test_labels = load_dataset()

    data_shape = training_data.shape[1:]

    activation_function = Sigmoid()
    cnn = CNN(
        [
            CR(5, 3, Adam(0.001), (1, 50, 50)),
            CR(3, 3, Adam(0.001), (5, 48, 48)),
            Flatten(),
            FullyConnected(
                6348,
                100,
                activation_function,
                Adam(0.001),
            ),
            FullyConnected(100, 50, activation_function, Adam(0.001)),
            FullyConnected(50, 5, activation_function, Adam(0.001)),
            FullyConnected(5, 1, Sigmoid(), Adam(0.001)),
        ]
    )

    loss_per_epoch = cnn.train(training_data, training_labels, 5, 10)

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
