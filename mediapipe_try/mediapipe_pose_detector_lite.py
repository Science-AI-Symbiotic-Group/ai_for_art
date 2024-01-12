import mediapipe as mp
import cv2
import numpy as np


## initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


mp_styles = mp.solutions.drawing_styles

landmark_annotations = mp_styles.get_default_pose_landmarks_style()

custom_color_1 = (121, 111, 22)
landmark_annotations= mp_drawing.DrawingSpec(
    color=custom_color_1, thickness=4, circle_radius=10)




pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


cap = cv2.VideoCapture(1)


while cap.isOpened():
    # read fram
     _, frame = cap.read()

     # resize the frame for portrait video
     # frame = cv2.resize(frame, (350, 600))
     # convert to RGB
     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

     # process the frame for pose detection
     pose_results = pose.process(frame_rgb)
     # print(pose_results.pose_landmarks)
     black_bkgrnd=np.zeros_like(frame_rgb)

     # draw skeleton on the frame

     mp_drawing.draw_landmarks(black_bkgrnd, pose_results.pose_landmarks, 
                              mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=landmark_annotations)
     # display the frame
     cv2.imshow('Output', black_bkgrnd)
     
     
     if cv2.waitKey(1) == ord('q'):
          break
               
cap.release()
cv2.destroyAllWindows()
