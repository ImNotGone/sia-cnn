import matplotlib.pyplot as plt

def visualize_first_layer_filters(cnn):
    filters = cnn.get_filters()
    first_layer_filters = filters[0]

    for filter_index, filter in enumerate(first_layer_filters):
        # Only one channel in first layer
        filter = filter[0]

        plt.imshow(filter, cmap='gray')
        for i in range(len(filter)):
            for j in range(len(filter[i])):
                value=filter[i,j]
                color= 'green' if value >=0 else 'red'
                plt.text(j,i, f'{value:.2f}', ha='center', va='center', color=color)
        plt.title('CNN Filter Visualization')
        plt.colorbar()
        plt.savefig(f'filter_first-layer_{filter_index}.png')
        plt.clf()

def visualize_feature_maps(cnn, data, label=""):
    feature_maps = cnn.get_feature_maps(data)

    for layer_index, layer_feature_maps in enumerate(feature_maps):
        for feature_map_index, feature_map in enumerate(layer_feature_maps):
            plt.imshow(feature_map, cmap='gray')
            plt.title(f'CNN Feature Map Visualization {label}')
            plt.colorbar()
            plt.savefig(f'feature_map_{layer_index}_{feature_map_index}_{label}.png')
            plt.clf()


def plot_errors_per_architecture(
    errors_per_architecture: dict[str, tuple[float, float]]
):
    print(errors_per_architecture.values())
    means = [error[0] for error in errors_per_architecture.values()]
    stds = [error[1] for error in errors_per_architecture.values()]

    plt.bar(
        [i + 1 for i in range(len(errors_per_architecture))],
        means,
        yerr=stds,
        capsize=5,
    )

    names = list(errors_per_architecture.keys())

    plt.xticks(
        [i + 1 for i in range(len(errors_per_architecture))],
        names,
    )

    plt.xlabel("Architecture")
    plt.ylabel("Error")

    plt.title("Mean Error per architecture (10 iterations)")

    plt.savefig("errors_per_architecture.png")
    plt.figure()
