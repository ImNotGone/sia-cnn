import asyncio
import concurrent.futures
import numpy as np
import json
from cnn import CNN
from utils.dataset_loader import load_dataset
from layers.convolutional import Convolutional
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from layers.pooling import Pooling
from layers.softmax import SM
from layers.utils.activation_functions import ReLU, Sigmoid
from layers.utils.optimization_methods import Adam, GradientDescent, Momentum
from utils.plots import plot_errors_per_architecture
import copy  

def create_layers_from_text(architecture_text, delta):
    activation_function = Sigmoid()  # Define your activation function here

    layers = []
    for layer_info in architecture_text:
        layer_type = layer_info[0]
        if layer_type == 'Convolutional':
            _, kernel_size, _, adam_params, input_shape = layer_info
            layers.append(Convolutional(*kernel_size, Adam(delta), input_shape))
        elif layer_type == 'Pooling':
            _, pool_shape = layer_info
            layers.append(Pooling(pool_shape))
        elif layer_type == 'Flatten':
            layers.append(Flatten())
        elif layer_type == 'FullyConnected':
            _, input_size, output_size = layer_info
            layers.append(FullyConnected(input_size, output_size, activation_function, Adam(delta)))
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    return layers

def train_and_calculate_error(cnn, training_data, training_labels, epochs, batch_size):
    return CNN(cnn).train(training_data, training_labels, epochs, batch_size)

async def architecture_test_async(architecture, iterations, training_data, training_labels, epochs, batch_size):
    mean_errors = []

    for _ in range(iterations):
        errors = []
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            tasks = [
                loop.run_in_executor(pool, train_and_calculate_error, architecture, training_data, training_labels, epochs, batch_size)
            ]
            for task in await asyncio.gather(*tasks):
                best_error = min(task)
                errors.append(best_error)

        mean_error = np.mean(errors)
        std_error = np.std(errors)
        mean_errors.append((mean_error, std_error))

    return errors

async def run_multiple_architectures2():
    delta = 0.001
    activation_function = Sigmoid()

    #architectures = [
    #    [
    #        Convolutional(5, 3, Adam(delta), (1, 50, 50)),
    #        Pooling((5, 48, 48)),
    #        Convolutional(3, 3, Adam(delta), (5, 24, 24)),
    #        Pooling((3, 22, 22)),
    #        Flatten(),
    #        FullyConnected(
    #            363,
    #            100,
    #            activation_function,
    #            Adam(delta),
    #        ),
    #        FullyConnected(100, 50, activation_function, Adam(delta)),
    #        FullyConnected(50, 5, activation_function, Adam(delta)),
    #        FullyConnected(5, 1, activation_function, Adam(delta)),

    #    ]
    #]
    
    architecture_text = [
        ['Convolutional', (5, 3), None, None, (1, 50, 50)],
        ['Pooling', (5, 48, 48)],
        ['Convolutional', (3, 3), None, None, (5, 24, 24)],
        ['Pooling', (3, 22, 22)],
        ['Flatten'],
        ['FullyConnected', 363, 100],
        ['FullyConnected', 100, 50],
        ['FullyConnected', 50, 5],
        ['FullyConnected', 5, 1]
    ]

    iterations = 5
    architectures = []
    for _ in range(iterations):  
        
        layers = create_layers_from_text(architecture_text, delta)
        architectures.append(layers)

    training_data, training_labels, _, _ = load_dataset()
    epochs = 3
    batch_size = 100
    
    tasks = []
    for idx, arch in enumerate(architectures, start=1):
        for _ in range(1):
            tasks.append(architecture_test_async(arch, 1, training_data, training_labels, epochs, batch_size))

    results = await asyncio.gather(*tasks)

    mean_errors_per_architecture = {f"cnn{i+1}": errors for i, errors in enumerate(results)}
    #plot_errors_per_architecture(mean_errors_per_architecture)
    print(mean_errors_per_architecture)
    with open("errors_architecture.json", "w") as f:
        json.dump(mean_errors_per_architecture, f)

# No funciona ninguna de las dos por ahora, haga lo que haga corren las mismas instancias de cnn
# y sigue entrenando las 2 base pero por muchas mas epoch
# Funciona si le pasas en architecture le pones 10 arquitecturas de cada una que queres probar
# y con una iteracion, te corre las 10 al mismo tiempo y son la misma pero tenes que copiarlas
# dsp lo corrijo para que funcione mas intuitivo

# el grafico no lo hace todavia
if __name__ == "__main__":
    asyncio.run(run_multiple_architectures2())

