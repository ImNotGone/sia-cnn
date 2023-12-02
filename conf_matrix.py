from cnn import CNN
from dataset_loader import load_dataset
from layers.cr import CR
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from layers.softmax import SM
from layers.utils.activation_functions import ReLU, Sigmoid
from layers.utils.optimization_methods import Adam, GradientDescent, Momentum

import matplotlib.pyplot as plt
import numpy as np


def main():
    training_data, training_labels, test_data, test_labels = load_dataset()

    data_shape = training_data.shape[1:]

    activation_function = Sigmoid()
    cnn = CNN(
        [
            CR(2, 3, Adam(0.001)),
            Flatten(),
            FullyConnected(
                (data_shape[0] - 2) * (data_shape[1] - 2) * 2,
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

    loss_per_epoch = cnn.train(training_data, training_labels, 2, 10)

    correct_squares=0
    false_triangles=0
    correct_triangles=0
    false_squares=0
        
    total_predictions = 0
    for data, label in zip(test_data, test_labels):
        sample = data.reshape(data.shape[0], data.shape[1], 1)
        output = cnn.forward_prop(sample)

        predicted = "square" if output < 0.5 else "triangle"
        actual = "square" if label == 0 else "triangle"
        
        if output < 0.5:
            if label == 0 :
                correct_squares+=1
            else:
                false_triangles+=1
        elif output > 0.5:
            if label == 1:
                correct_triangles+=1
            else:
                #print("++")
                false_squares+=1

        #print(f"Predicted: {predicted}, Output: {output}")
        #print(f"Actual: {actual}, Label: {label}")

        if predicted == actual:
            total_predictions += 1

    print("Accuracy: ", total_predictions / len(test_data))

    confusion_matrix_2x2 = np.array([[correct_squares, false_triangles],[false_squares, correct_triangles]])
    
    _ , ax = plt.subplots()
    
    
    cax = ax.matshow(confusion_matrix_2x2, cmap=plt.cm.Blues)

    for i in range(confusion_matrix_2x2.shape[0]):
        for j in range(confusion_matrix_2x2.shape[1]):
            ax.text(j, i, str(confusion_matrix_2x2[i, j]), va='center', ha='center', color='red')

    #los verticales osn los verdaderos
    plt.xticks(np.arange(2), ['Actual Square', 'Actual Triangle'])
    plt.yticks(np.arange(2), ['Predicted Square', 'Predicted Triangle'])

    
    plt.colorbar(cax)

    # Título del gráfico
    plt.title('Confusion Matrix')

    # Mostrar el gráfico
    plt.show()




if __name__ == "__main__":
    main()
