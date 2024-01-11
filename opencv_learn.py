import cv2
import numpy as np



#select the camera
cap= cv2.VideoCapture(2)

# start the camera

while True:

# select the frames of the camera
    
    ret, frame = cap.read()

    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    """
    # Adjust Canny edge detection thresholds for sensitivity
    low_threshold = 100  # You can experiment with different values
    high_threshold = 205  # You can experiment with different values
    edges = cv2.Canny(gray, low_threshold, high_threshold)

     # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a black background
    black_background = np.zeros_like(frame)

    cv2.drawContours(black_background, contours, -1, (0, 255, 0), 2)
    

    cv2.imshow('edges', edges)
    cv2.imshow('gray', gray)
    """
    #How to display the frame on the monitor
    cv2.imshow('Frames', frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()