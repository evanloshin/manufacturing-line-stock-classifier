# Load dependencies
import functions
import cv2
import matplotlib.pyplot as plt
import datetime


def main():

    # Define the webcam object
    cam = cv2.VideoCapture(1)

    while(True):

        # Capture video frame
        ret, frame = cam.read()

        # Display frame
        cv2.imshow('my webcam', frame)

        # Quit capture mode
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save frame
        if cv2.waitKey(60) & 0xFF == ord('s'):
            timestamp = datetime.datetime.now()
            filename = '/Users/evanloshin/Documents/Udacity/manufacturing-line-stock-classifier/images/' + str(timestamp) + '.png'
            plt.imsave(filename, frame)
            print('Save successful: ' + str(timestamp))

    # When everything's done, release the capture
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
