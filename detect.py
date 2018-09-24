# Load dependencies
import functions
import cv2
from keras.models import load_model
import pickle


def main():

    # Load one-hot-encoding matrix
    labeler = functions.load_object('one-hot-matrix.pkl')

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
        logits = model.predict(image, batch_size=1)

        # Decode logits
        result = labeler.inverse_transform(logits)


if __name__ == '__main__':
    main()
