# Load dependencies
import cv2
from sklearn import preprocessing
from scipy.ndimage import rotate
import numpy as np
import random


def preprocess(image):
    """
    Grayscale and normalize image to improve classifier
    results.

    :param image: numpy array of image pixels
    :return: conditioned numpy array
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    normal = preprocessing.normalize(grayscale)
    return normal


def augment_dataset(images, labels):
    """
    Increase quantity of images in the training data set by
    combination of rotating and flipping.

    :param images: numpy array of images
    :param labels: numpy array of labels
    :return: numpy array of images and augmented images, corresponding numpy array of labels
    """
    for img, label in zip(images, labels):
        # Rotate original image
        rotated = rotate_random(img)
        # Flip original image
        flipped = np.fliplr(img)
        # Rotate flipped image
        rotated_flipped = rotate_random(flipped)
        # Add two new samples to training data set
        images = np.append(images, [rotated, rotated_flipped], axis=0)
        labels = np.append(labels, [label, label])
    # return images and labels
    return images, labels


def rotate_random(img):
    """
    Rotates image by a randomly chosen angle between 0-180 degrees
    and crops result to maintain original shape.

    :param img: original image
    :return: rotated image
    """
    dims_orig = img.shape
    angle = random.random() * 180
    rotated = rotate(img, angle)
    dims_rtd = rotated.shape
    lower_width = (dims_rtd[1] - dims_orig[1]) // 2
    upper_width = dims_orig[1] + lower_width
    lower_height = (dims_rtd[0] - dims_orig[0]) // 2
    upper_height = dims_orig[0] + lower_height
    cropped = rotated[lower_height:upper_height, lower_width:upper_width]
    return cropped
