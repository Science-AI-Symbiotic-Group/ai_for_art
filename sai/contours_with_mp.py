import cv2
import numpy as np
import random
import mediapipe as mp


def draw_lines_on_limbs(frame,result):


    ## initialize pose estimator
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    


    mp_styles = mp.solutions.drawing_styles

    landmark_annotations = mp_styles.get_default_pose_landmarks_style()

    custom_color_1 = (0, 255, 128)


    # NOSE LANDMARKS
    #nose_x = result.pose_landmarks.landmark[mp]

    landmark_annotations = mp_drawing.DrawingSpec(
    color=custom_color_1, thickness=4, circle_radius=10)



    mp_drawing.draw_landmarks(frame, result.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_annotations)



    return frame


def add_random_noise(image, intensity=25):
    noisy_image = image.copy()
    noise = np.random.randint(-intensity, intensity + 1, noisy_image.shape)
    noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def main():
    

    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,num_poses=20)


    
    # Open a connection to the webcam (camera index 0 by default)
    cap = cv2.VideoCapture("videos/4.mp4")
    colorChangedTime = 0
    color = (128,255,0) # Setting the color for the edges



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
        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)


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

        # Generate random Gaussian noise
        

        cv2.drawContours(contour_frame, contours, -1, color, 1)

        contour_frame = draw_lines_on_limbs(contour_frame,pose_results)

        noisy_img = add_random_noise(contour_frame,30)
        
        # Display the frame with contours on a dark background
        cv2.imshow("Contours on Dark Background", noisy_img)

        
        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
