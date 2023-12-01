
from cnn import CNN
from dataset_loader import load_dataset
from layers.cr import CR
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from layers.softmax import SM
from layers.utils.activation_functions import Sigmoid
from layers.utils.optimization_methods import Adam, GradientDescent


def main():

    training_data, training_labels, test_data, test_labels = load_dataset()

    data_shape = training_data.shape[1:]

    optimization_method = GradientDescent(0.01)
    activation_function = Sigmoid()

    conv_layer = CR(5, 3, optimization_method)
    flatten_layer = Flatten()
    fully_connected_layer = FullyConnected((data_shape[0] - 2) * (data_shape[1] - 2) * 5, 5, activation_function, optimization_method)
    softmax_layer = SM(5, 2, optimization_method)

    cnn = CNN([conv_layer, flatten_layer, fully_connected_layer, softmax_layer])

    loss_per_epoch = cnn.train(training_data, training_labels, 5, 10)

    for data, label in zip(test_data, test_labels):
        output = cnn.forward_prop(data)

        predicted = "square" if output[0] > output[1] else "triangle"
        actual = "square" if label[0] > label[1] else "triangle"

        print("Predicted: ", predicted)
        print("Actual: ", actual)
        print()

    print("Training finished")

if __name__ == "__main__":
    main()
