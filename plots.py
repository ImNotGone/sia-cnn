import matplotlib.pyplot as plt

def visualize_filters(cnn):
    filters = cnn.get_filters()

    for layer_index, layer_filters in enumerate(filters):
        for filter_index, filter in enumerate(layer_filters):
            plt.imshow(filter, cmap='gray')
            plt.title('CNN Filter Visualization')
            plt.colorbar()
            plt.savefig(f'filter_{layer_index}_{filter_index}.png')
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

