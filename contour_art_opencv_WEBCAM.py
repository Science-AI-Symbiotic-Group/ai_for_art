import cv2
import numpy as np

def main():
    # Open a connection to the webcam (camera index 0 by default)
    cap = cv2.VideoCapture(2)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the webcam.")
        return

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error: Couldn't read a frame.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Adjust Canny edge detection thresholds for sensitivity
        low_threshold = 100  # You can experiment with different values
        high_threshold = 205  # You can experiment with different values
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Find contours in the edged image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a black background
        black_background = np.zeros_like(frame)

        # Draw contours on the black background
        contour_frame = black_background.copy()
        cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)

        # Display the frame with contours on a dark background
        cv2.imshow("Contours on Dark Background", contour_frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
