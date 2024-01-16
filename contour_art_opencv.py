import cv2
import numpy as np
cap=cv2.VideoCapture('video2.mp4')   #replace with webcam number
# Get the video details (width, height, frames per second)
width = cap.get(3)
height = cap.get(4)
fps = cap.get(5)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video2.avi', fourcc, 25.0, (848, 848))


while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame is read successfully
    if not ret:
        print("Error: Couldn't read a frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray=cv2.medianBlur(gray,1)
    gray = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=5) 

    # Adjust Canny edge detection thresholds for sensitivity
    low_threshold = 70 # You can experiment with different values
    high_threshold = 150  # You can experiment with different values
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black background
    black_background = np.zeros_like(frame)

    # Draw contours on the black background
    contour_frame = black_background.copy()
    cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)
    out.write(contour_frame)

    # Display the frame with contours on a dark background
    cv2.imshow("Contours on Dark Background", contour_frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
