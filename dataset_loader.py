import cv2
import numpy as np
import os

# TODO check if we have to set the cwd
# parameters
img_size = 50  # TODO check this
train_ratio = 0.8  # TODO check this


def load_dataset():
    folders = ["data/shapes/square", "data/shapes/triangle"]
    labels = []
    images = []

    for folder in folders:
        for path in os.listdir(os.getcwd() + "/" + folder):
            img = cv2.imread(folder + "/" + path, 0)
            images.append(cv2.resize(img, (img_size, img_size)))
            labels.append(folders.index(folder))

    # Separate data into training sets and testing sets
    train_qty = int(len(images) * train_ratio)

    # Make labels one-hot encoded,
    # e.g. 0 -> [1, 0], 1 -> [0, 1]
    labels = np.eye(len(folders))[labels]

    train_images = images[:train_qty]
    train_labels = labels[:train_qty]

    test_images = images[train_qty:]
    test_labels = labels[train_qty:]

    # Training set first
    train_images = np.array(train_images)
    train_images = train_images.astype("float32")
    train_images /= 255

    # Testing set second
    test_images = np.array(test_images)
    test_images = test_images.astype("float32")
    test_images /= 255

    return train_images, train_labels, test_images, test_labels
