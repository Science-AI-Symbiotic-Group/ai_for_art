import cv2
import numpy as np
import random
from scipy.ndimage import binary_erosion
from scipy.ndimage import gaussian_filter

def add_random_noise(image, intensity=25):
    noisy_image = image.copy()
    noise = np.random.randint(-intensity, intensity + 1, noisy_image.shape)
    noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def main():
    # Open a connection to the webcam (camera index 0 by default)
    cap = cv2.VideoCapture(1)
    colorChangedTime = 0
    color = (0,0,255) # Setting the color for the edges at the start



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
        low_threshold = 50  # You can experiment with different values
        high_threshold = 150  # You can experiment with different values
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Find contours in the edged image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a black background
        black_background = np.zeros_like(frame)

        # Draw contours on the black background
        contour_frame = black_background.copy()

        # Generate random Gaussian noise
        #noisy_img = contour_frame
        
        
        color_list =[(0,0,255),(64,0,215),] # a list of some color codes in BGR format
        if colorChangedTime < 60: # if the last time the color was updated was less than 30 frames ago, dont do anything
            color = color
        else:
            color = random.choice(color_list) # if color was updated more than 30 frames ago (runs on the 31'st frame), pick a new color from the list and change it
            colorChangedTime = 0
        

        cv2.drawContours(contour_frame, contours, -1, color, 1)

        noisy_img = add_random_noise(contour_frame,30)
        
        # Display the frame with contours on a dark background
        cv2.imshow("Contours on Dark Background", noisy_img)
        
        colorChangedTime = colorChangedTime + 1
        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
