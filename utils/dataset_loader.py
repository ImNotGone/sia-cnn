import cv2
import numpy as np
import os

# TODO check if we have to set the cwd
# parameters
img_size = 200  # TODO check this
train_ratio = 0.8  # TODO check this
samples_per_class = 100  # TODO check this


def load_dataset(img_size=img_size, train_ratio=train_ratio, samples_per_class=samples_per_class):
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

    # Reduce amount of data
    squares = []
    triangles = []
    for image, label in zip(images, labels):
        if label == 0:
            squares.append(image)
        else:
            triangles.append(image)
    images = squares[:samples_per_class] + triangles[:samples_per_class]
    labels = [0] * samples_per_class + [1] * samples_per_class

    sqare_qty = labels.count(0)
    triangle_qty = labels.count(1)
    # Separate data into training sets and testing sets
    train_qty = int(len(images) * train_ratio)
    square_train_qty = int(train_qty * sqare_qty / (sqare_qty + triangle_qty))
    triangle_train_qty = train_qty - square_train_qty

    print(f"There are {sqare_qty} squares and {triangle_qty} triangles")

    train_images = images[:square_train_qty] + images[samples_per_class : samples_per_class + triangle_train_qty]
    train_labels = labels[:square_train_qty] + labels[samples_per_class : samples_per_class + triangle_train_qty]
    test_images = images[square_train_qty : samples_per_class] + images[samples_per_class + triangle_train_qty :]
    test_labels = labels[square_train_qty : samples_per_class] + labels[samples_per_class + triangle_train_qty :]

    print(f"There are {len(train_images)} training images and {len(test_images)} testing images")


    # Training set first
    train_images = np.array(train_images)
    train_images = train_images.astype("float32")
    train_images /= 255

    # Testing set second
    test_images = np.array(test_images)
    test_images = test_images.astype("float32")
    test_images /= 255

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels
