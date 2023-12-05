import matplotlib.pyplot as plt
import numpy as np


# Create plot dir if it doesn't exist
import os

plot_dir = "results"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


def visualize_first_layer_filters(cnn):
    filters = cnn.get_filters()
    first_layer_filters = filters[0]

    for filter_index, filter in enumerate(first_layer_filters):
        # Only one channel in first layer
        filter = filter[0]

        plt.imshow(filter, cmap="gray")
        for i in range(len(filter)):
            for j in range(len(filter[i])):
                value = filter[i, j]
                color = "green" if value >= 0 else "red"
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", color=color)
        plt.title("CNN Filter Visualization")
        plt.colorbar()
        plt.savefig(f"{plot_dir}/filter_first-layer_{filter_index}.png")
        plt.clf()


def visualize_feature_maps(cnn, data, label=""):
    feature_maps = cnn.get_feature_maps(data)

    for layer_index, layer_feature_maps in enumerate(feature_maps):
        for feature_map_index, feature_map in enumerate(layer_feature_maps):
            plt.imshow(feature_map, cmap="gray")
            plt.title(f"CNN Feature Map Visualization {label}")
            plt.colorbar()
            plt.savefig(f"{plot_dir}/feature_map_{layer_index}_{feature_map_index}_{label}.png")
            plt.clf()


def plot_errors_per_architecture(
    errors_per_architecture: dict[str, tuple[float, float]],
    path="errors_per_architecture.png",
):
    names = list(errors_per_architecture.keys())
    means = [error[0] for error in errors_per_architecture.values()]
    stds = [error[1] for error in errors_per_architecture.values()]

    plt.bar(
        [i + 1 for i in range(len(errors_per_architecture))],
        means,
        yerr=stds,
        capsize=5,
    )


    plt.xticks(
        [i + 1 for i in range(len(errors_per_architecture))],
        names,
        rotation=45,  # Rotate x-tick labels by 45 degrees
        ha='right',  # Align rotated labels to the right
        fontsize=8,  # Adjust font size
    )

    plt.xlabel("Architecture")
    plt.ylabel("Error")

    # Do not show negative values on y ticks
    max_ytick = max(means) + max(stds)
    plt.yticks(np.arange(0, max_ytick + 0.1, 0.1))

    plt.title("Mean Error per architecture (10 iterations)")

    # Increase figure width for more space
    fig = plt.gcf()
    fig.set_size_inches(10, 5)

    plt.tight_layout()  # Adjust layout to prevent clipping of labels

    plt.savefig(f"{plot_dir}/{path}")
    plt.clf()

def plot_errors_per_epoch(errors_per_epoch: list[float]):
    plt.plot(errors_per_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Error")

    plt.title("Error per epoch")

    plt.savefig(f"{plot_dir}/errors_per_epoch.png")
    plt.clf()


def plot_confusion_matrix(predictions: list[tuple[str, str, float, float]]):
    square_predictions = [
        prediction for prediction in predictions if prediction[0] == "square"
    ]
    correct_squares = len(
        [prediction for prediction in square_predictions if prediction[1] == "square"]
    )
    false_squares = len(square_predictions) - correct_squares

    triangle_predictions = [
        prediction for prediction in predictions if prediction[0] == "triangle"
    ]
    correct_triangles = len(
        [
            prediction
            for prediction in triangle_predictions
            if prediction[1] == "triangle"
        ]
    )
    false_triangles = len(triangle_predictions) - correct_triangles
    
    confusion_matrix_2x2 = np.array([[correct_squares, false_squares], [false_triangles, correct_triangles]])

    _, ax = plt.subplots()

    cax = ax.matshow(confusion_matrix_2x2, cmap=plt.cm.Blues)

    for i in range(confusion_matrix_2x2.shape[0]):
        for j in range(confusion_matrix_2x2.shape[1]):
            color = "white" if i == j else "red"
            ax.text(j, i, str(confusion_matrix_2x2[i, j]), va='center', ha='center', color=color)

    #los verticales osn los verdaderos
    plt.xticks(np.arange(2), ['Actual Square', 'Actual Triangle'])
    # yticks centrados en el medio de la celda, por eso el 0.5
    plt.yticks(np.arange(2), ['Predicted Square', 'Predicted Triangle'], rotation=90, va='center', ha='center')



    
    plt.colorbar(cax)

    plt.title('Confusion Matrix')

    plt.savefig(f"{plot_dir}/confusion_matrix.png")
    plt.clf()
