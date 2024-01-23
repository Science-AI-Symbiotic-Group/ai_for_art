import mediapipe as mp
import cv2
import numpy as np


## initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture('video4.mp4')
# Get the video details (width, height, frames per second)



# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video4.avi', fourcc, 25.0, (608, 1080))


while cap.isOpened():
    # read frame
    _, frame = cap.read()
    try:
         # resize the frame for portrait video
         # frame = cv2.resize(frame, (350, 600))
         # convert to RGB
         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
         # process the frame for pose detection
         pose_results = pose.process(frame_rgb)

         black_background=np.zeros_like(frame)
         # print(pose_results.pose_landmarks)
         
         # draw skeleton on the frame
         mp_drawing.draw_landmarks(black_background, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
         # display the frame
         cv2.imshow('Output', black_background)
         out.write(black_background)

    except:
         break
    
    if cv2.waitKey(1) == ord('q'):
          break
          
cap.release()
cv2.destroyAllWindows()
