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

    # Shuffle data
    data = list(zip(images, labels))
    np.random.shuffle(data)
    images = [d[0] for d in data]
    labels = [d[1] for d in data]

    # Only keep first 100 images and last 100 images
    images = images[:100] + images[-100:]
    labels = labels[:100] + labels[-100:]

    sqare_qty = labels.count(0)
    triangle_qty = labels.count(1)
    print(f"Square qty: {sqare_qty}")
    print(f"Triangle qty: {triangle_qty}")

    # Separate data into training sets and testing sets
    train_qty = int(len(images) * train_ratio)

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

    np.array(train_labels)
    np.array(test_labels)

    return train_images, train_labels, test_images, test_labels
