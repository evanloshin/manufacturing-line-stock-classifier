# Load dependencies
import functions
import cv2
from keras.models import load_model


def main():

    # Load the trained model
    model = load_model('model.h5')

    # Define the webcam object
    cam = cv2.VideoCapture(0)

    while(True):

        # Capture video frame
        ret, frame = cam.read()

        # Preprocess frame
        image = functions.preprocess(frame)

        # Predict object in frame
        result = model.predict(image, batch_size=1)


if __name__ == '__main__':
    main()
