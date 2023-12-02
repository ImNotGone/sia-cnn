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
def calculate_roc_curve(true_labels, predictions):
    
    sorted_predictions, sorted_labels = zip(*sorted(zip(predictions, true_labels), reverse=True))

    total_positive = sum(true_labels)
    total_negative = len(true_labels) - total_positive

    tpr_values = []
    fpr_values = []

    tp = 0
    fp = 0

    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / total_positive
        fpr = fp / total_negative

        tpr_values.append(tpr)
        fpr_values.append(fpr)

    return fpr_values, tpr_values

def calculate_auc(fpr_values, tpr_values):
    auc_value = 0
    for i in range(1, len(fpr_values)):
        auc_value += (fpr_values[i] - fpr_values[i - 1]) * tpr_values[i]
    return auc_value

def main():
    training_data, training_labels, test_data, test_labels = load_dataset()

    data_shape = training_data.shape[1:]

    activation_function = Sigmoid()
    cnn = CNN(
        [
            CR(1, 3, Adam(0.001)),
            Flatten(),
            FullyConnected(
                (data_shape[0] - 2) * (data_shape[1] - 2) * 1,
                5,
                activation_function,
                Adam(0.001),
            ),
            FullyConnected(5, 1, Sigmoid(), Adam(0.001)),

        ]
    )
    
    cnn2 = CNN(
        [
            CR(1, 3, Adam(0.0001)),
            Flatten(),
            FullyConnected(
                (data_shape[0] - 2) * (data_shape[1] - 2) * 1,
                5,
                activation_function,
                Adam(0.0001),
            ),
            FullyConnected(5, 1, Sigmoid(), Adam(0.0001)),

        ]
    )
    
    cnn3 = CNN(
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

    loss_per_epoch = cnn.train(training_data, training_labels, 1, 10)
    loss_per_epoch2 = cnn2.train(training_data, training_labels, 2, 10)
    loss_per_epoch2 = cnn3.train(training_data, training_labels, 2, 10)


    true_labels_model1 = []  
    predictions_model1 = [] 
    
    true_labels_model2 = []  
    predictions_model2 = []  
    
    true_labels_model3 = []  
    predictions_model3 = []  

    for data, label in zip(test_data, test_labels):
        sample = data.reshape(data.shape[0], data.shape[1], 1)
        output_model1 = cnn.forward_prop(sample)  

        true_labels_model1.append(label)
        predictions_model1.append(output_model1)

   
    for data, label in zip(test_data, test_labels):
        sample = data.reshape(data.shape[0], data.shape[1], 1)
        output_model2 = cnn2.forward_prop(sample)  

        true_labels_model2.append(label)
        predictions_model2.append(output_model2)
        
    for data, label in zip(test_data, test_labels):
        sample = data.reshape(data.shape[0], data.shape[1], 1)
        output_model3 = cnn3.forward_prop(sample) 

        true_labels_model3.append(label)
        predictions_model3.append(output_model3)

    true_labels_model1 = np.array(true_labels_model1)
    predictions_model1 = np.array(predictions_model1).flatten()

    true_labels_model2 = np.array(true_labels_model2)
    predictions_model2 = np.array(predictions_model2).flatten()
    
    true_labels_model3 = np.array(true_labels_model3)
    predictions_model3 = np.array(predictions_model3).flatten()
    

    fpr_values_model1, tpr_values_model1 = calculate_roc_curve(true_labels_model1, predictions_model1)
    auc_value_model1 = calculate_auc(fpr_values_model1, tpr_values_model1)

    fpr_values_model2, tpr_values_model2 = calculate_roc_curve(true_labels_model2, predictions_model2)
    auc_value_model2 = calculate_auc(fpr_values_model2, tpr_values_model2)
    
    fpr_values_model3, tpr_values_model3 = calculate_roc_curve(true_labels_model3, predictions_model3)
    auc_value_model3 = calculate_auc(fpr_values_model3, tpr_values_model3)

    plt.figure()
    plt.plot(fpr_values_model1, tpr_values_model1, color='blue', lw=2, label=f'Model 1 (AUC = {auc_value_model1:.2f})')
    plt.plot(fpr_values_model2, tpr_values_model2, color='red', lw=2, label=f'Model 2 (AUC = {auc_value_model2:.2f})')
    plt.plot(fpr_values_model3, tpr_values_model3, color='green', lw=2, label=f'Model 3 (AUC = {auc_value_model3:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.show()
    
    
    
    # Investigue mas y no se que tan bueno esta este grafico
    # Es mas cuando te interesa minimizar los falsos positivos o los falsos negativos
    # Lo podriamos poner si decimos que nos queremos centrar en identificar mejor los
    # cuadrados sin que nos importen los triangulos
    
    

    




if __name__ == "__main__":
    main()
