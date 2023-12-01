import cv2, numpy as np, os

#TODO check if we have to set the cwd
#parameters
img_size = 50 #TODO check this
train_qty= 30 #change this value to alter the amount of images used for training in the set

def load_dataset():
    #get data TODO complete folders array with the name of the folder where the images are
    folders, labels, images = [], [], []
    for folder in folders:
        for path in os.listdir(os.getcwd()+'/'+folder):
            img = cv2.imread(folder+'/'+path,0)
            images.append(cv2.resize(img, (img_size, img_size)))
            labels.append(folders.index(folder))

    #Separate data into training sets and testing sets
    i=0
    train_images, train_labels, test_images, test_labels = [],[],[],[]
    for image, label in zip(images, labels):
        if i<train_qty:
            train_images.append(image)
            train_labels.append(label)
            i+=1
        else:
            test_images.append(image)
            test_labels.append(label)

    #We flatten the data
    #Training set first
    train_images = np.array(train_images)
    train_images = train_images.astype('float32')
    train_images /=255

    #Testing set second
    test_images = np.array(test_images)
    test_images = test_images.astype('float32')
    test_images /=255

    return train_images, train_labels, test_images, test_labels