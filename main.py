from cnn import CNN
from dataset_loader import load_dataset
from layers.cr import CR
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from layers.softmax import SM
from layers.utils.activation_functions import ReLU, Sigmoid
from layers.utils.optimization_methods import Adam, GradientDescent, Momentum


def main():
    training_data, training_labels, test_data, test_labels = load_dataset()

    data_shape = training_data.shape[1:]

    activation_function = Sigmoid()
    cnn = CNN(
        [
            CR(5, 3, Adam(0.001)),
            Flatten(),
            FullyConnected(
                (data_shape[0] - 2) * (data_shape[1] - 2) * 5,
                1000,
                activation_function,
                Adam(0.001),
            ),
            FullyConnected(1000, 500, activation_function, Adam(0.001)),
            FullyConnected(500, 100, activation_function, Adam(0.001)),
            FullyConnected(100, 50, activation_function, Adam(0.001)),
            FullyConnected(50, 5, activation_function, Adam(0.001)),
            FullyConnected(5, 1, Sigmoid(), Adam(0.001)),
        ]
    )

    loss_per_epoch = cnn.train(training_data, training_labels, 5, 10)

    total_predictions = 0
    for data, label in zip(test_data, test_labels):
        output = cnn.forward_prop(data)

        predicted = "square" if output < 0.5 else "triangle"
        actual = "square" if label == 0 else "triangle"

        print(f"Predicted: {predicted}, Output: {output}")
        print(f"Actual: {actual}, Label: {label}")

        if predicted == actual:
            total_predictions += 1
        print()

    print("Accuracy: ", total_predictions / len(test_data))

    print("Training finished")


if __name__ == "__main__":
    main()
