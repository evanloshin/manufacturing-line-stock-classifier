# Load dependencies
import functions
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage import imread
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from sklearn import preprocessing

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 10
VALIDATION_SPLIT = 0.3

# Model architecture
model = Sequential()
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', name='first_convolution', input_shape=(200, 200, 3)))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu', name='second_convolution'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu', name='third_convolution'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


def main():

    # Load training data from csv
    filename = 'train_data.csv'
    raw_data = np.loadtxt(filename, dtype=np.str, delimiter=",")
    image_paths = raw_data[:, 0]
    labels = raw_data[:, 1]

    # One-hot encode labels
    lb = preprocessing.LabelBinarizer()
    one_hot_labels = lb.fit_transform(labels)
    # Save transformation matrix
    functions.save_object(lb, 'one-hot-matrix.pkl')

    # Load training images from directory
    images = []
    for path in image_paths:
        images.append(imread(path))
    images = np.array(images)

    # pre-process images
    images = np.array([functions.preprocess(img) for img in images])

    # Split data into training and validation sets
    x_train, x_valid, y_train, y_valid = train_test_split(images, one_hot_labels, test_size=VALIDATION_SPLIT)

    # Augment training set with rotated and flipped images
    x_train, y_train = functions.augment_dataset(x_train, y_train)

    # Compile and run the neural network model
    # model.compile(loss='mse', optimizer='adam')
    # model.fit(x_train, y_train,
    #           batch_size=BATCH_SIZE,
    #           epochs=EPOCHS,
    #           validation_data=(x_valid, y_valid))
    # model.save('model.h5')


if __name__ == '__main__':
    main()
