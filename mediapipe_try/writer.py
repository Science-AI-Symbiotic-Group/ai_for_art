import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os

HEIGHT = 480
WIDTH = 640
model_path = 'D:/VsCodeProjects/Python/Github Projects/video_to_doodle/ai_for_art/mediapipe_try/pose_landmarker_lite.task'
frame_timestamp = 1


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])

            
    
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def draw_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # LIST OF IMPORTANT LANDMARK LOCATIONS 
    # 12 - 11 - 24 - 23 = for Torso corners | Could make a Square On these points
    # 12 - 14 - 16 - 18 - 20 = for Right Hand  | 11 - 13 - 15 - 17 - 19 = For Left Hand
    # 10 - 9 = Mouth
    # 24 - 26 - 28 = for Right Leg | 23 - 25 - 27 = for Left Leg

    # MAKES LINES FOR TORSO
    Torso_Color = (0,0,0) # in RGB Values

    TorsoLeftTop_Cords = (int(pose_landmarks_list[0][12].x * WIDTH), int(pose_landmarks_list[0][12].y * HEIGHT) )
    TorsoRightBottom_Cords = (int(pose_landmarks_list[0][23].x * WIDTH), int(pose_landmarks_list[0][23].y * HEIGHT ))
    TorsoRightTop_Cords = (int(pose_landmarks_list[0][11].x * WIDTH), int(pose_landmarks_list[0][11].y * HEIGHT ))
    TorsoLeftBottom_Cords = (int(pose_landmarks_list[0][24].x * WIDTH), int(pose_landmarks_list[0][24].y * HEIGHT ))

    cv2.line(annotated_image, TorsoLeftTop_Cords, TorsoRightTop_Cords, Torso_Color, thickness=10)
    cv2.line(annotated_image, TorsoRightTop_Cords, TorsoRightBottom_Cords, Torso_Color, thickness=10)
    cv2.line(annotated_image, TorsoRightBottom_Cords, TorsoLeftBottom_Cords, Torso_Color, thickness=10)
    cv2.line(annotated_image, TorsoLeftBottom_Cords, TorsoLeftTop_Cords, Torso_Color, thickness=10)

    # MAKE LINES FOR RIGHT HAND

    Hand_Color = (0,0,0)

    RightHandStart_Cords = (int(pose_landmarks_list[0][12].x * WIDTH), int(pose_landmarks_list[0][12].y * HEIGHT) )
    RightHand2_Cords = (int(pose_landmarks_list[0][14].x * WIDTH), int(pose_landmarks_list[0][14].y * HEIGHT) )
    RightHand3_Cords = (int(pose_landmarks_list[0][16].x * WIDTH), int(pose_landmarks_list[0][16].y * HEIGHT) )
    RightHandEnd_Cords = (int(pose_landmarks_list[0][18].x * WIDTH), int(pose_landmarks_list[0][18].y * HEIGHT) )
    
    cv2.line(annotated_image, RightHandStart_Cords, RightHand2_Cords, Hand_Color, thickness=10)
    cv2.line(annotated_image, RightHand2_Cords, RightHand3_Cords, Hand_Color, thickness=10)
    cv2.line(annotated_image, RightHand3_Cords, RightHandEnd_Cords, Hand_Color, thickness=10)


    # MAKE LINES FOR LEFT HAND

    LeftHandStart_Cords = (int(pose_landmarks_list[0][11].x * WIDTH), int(pose_landmarks_list[0][11].y * HEIGHT) )
    LeftHand2_Cords = (int(pose_landmarks_list[0][13].x * WIDTH), int(pose_landmarks_list[0][13].y * HEIGHT) )
    LeftHand3_Cords = (int(pose_landmarks_list[0][15].x * WIDTH), int(pose_landmarks_list[0][15].y * HEIGHT) )
    LeftHandEnd_Cords = (int(pose_landmarks_list[0][17].x * WIDTH), int(pose_landmarks_list[0][17].y * HEIGHT) )
    
    cv2.line(annotated_image, LeftHandStart_Cords, LeftHand2_Cords, Hand_Color, thickness=10)
    cv2.line(annotated_image, LeftHand2_Cords, LeftHand3_Cords, Hand_Color, thickness=10)
    cv2.line(annotated_image, LeftHand3_Cords, LeftHandEnd_Cords, Hand_Color, thickness=10)

    # MAKE LINES FOR THE MOUTH

    Mouth_Color = (0,0,0)

    MouthStart_Cords = (int(pose_landmarks_list[0][10].x * WIDTH), int(pose_landmarks_list[0][10].y * HEIGHT) )
    MouthEnd_Cords = (int(pose_landmarks_list[0][9].x * WIDTH), int(pose_landmarks_list[0][9].y * HEIGHT) )

    cv2.line(annotated_image, MouthStart_Cords, MouthEnd_Cords, Mouth_Color, thickness=10)

    # MAKE A CIRCLE FOR THE FACE

    Circle_Color = (0,0,0) 

    FaceCenter_Cords = (int(pose_landmarks_list[0][0].x * WIDTH), int(pose_landmarks_list[0][0].y * HEIGHT) )

    cv2.circle(annotated_image, FaceCenter_Cords, 60, Circle_Color, thickness=5, lineType=8, shift=0)

    # MAKE EYE

    Eye_Color = (0,0,0) 

    RightEye_Cords = (int(pose_landmarks_list[0][5].x * WIDTH), int(pose_landmarks_list[0][5].y * HEIGHT) )

    cv2.circle(annotated_image, RightEye_Cords, 10, Eye_Color, thickness=3, lineType=8, shift=0)

    LeftEye_Cords = (int(pose_landmarks_list[0][2].x * WIDTH), int(pose_landmarks_list[0][2].y * HEIGHT) )

    cv2.circle(annotated_image, LeftEye_Cords, 10, Eye_Color, thickness=3, lineType=8, shift=0)

    # MAKE LINES FOR RIGHT LEG

    Leg_Color = (0,0,0)

    RightLegStart_Cords = (int(pose_landmarks_list[0][24].x * WIDTH), int(pose_landmarks_list[0][24].y * HEIGHT) )
    RightLeg2_Cords = (int(pose_landmarks_list[0][26].x * WIDTH), int(pose_landmarks_list[0][26].y * HEIGHT) )
    RightLegEnd_Cords = (int(pose_landmarks_list[0][28].x * WIDTH), int(pose_landmarks_list[0][28].y * HEIGHT) )
    
    cv2.line(annotated_image, RightLegStart_Cords, RightLeg2_Cords, Leg_Color, thickness=10)
    cv2.line(annotated_image, RightLeg2_Cords, RightLegEnd_Cords, Leg_Color, thickness=10)


    # MAKE LINES FOR LEFT LEG

    LeftLegStart_Cords = (int(pose_landmarks_list[0][23].x * WIDTH), int(pose_landmarks_list[0][23].y * HEIGHT) )
    LeftLeg2_Cords = (int(pose_landmarks_list[0][25].x * WIDTH), int(pose_landmarks_list[0][25].y * HEIGHT) )
    LeftLegEnd_Cords = (int(pose_landmarks_list[0][27].x * WIDTH), int(pose_landmarks_list[0][27].y * HEIGHT) )
    
    cv2.line(annotated_image, LeftLegStart_Cords, LeftLeg2_Cords, Leg_Color, thickness=10)
    cv2.line(annotated_image, LeftLeg2_Cords, LeftLegEnd_Cords, Leg_Color, thickness=10)


    

    return annotated_image



options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=20,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
)



blank_image = np.ones((HEIGHT,WIDTH,3),np.uint8)
blank_image = 255* blank_image

cap = cv2.VideoCapture("videos/4.mp4")
with PoseLandmarker.create_from_options(options) as landmarker:
    while True:        
        ret ,frame = cap.read()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = landmarker.detect_for_video(mp_image,frame_timestamp)
        frame_timestamp = frame_timestamp + 1
        frame_drawn = draw_on_image(blank_image,detection_result=detection_result)
        frame_landmarks = draw_landmarks_on_image(blank_image,detection_result=detection_result)
        
        cv2.imshow("animation",frame_drawn)
        cv2.imshow("landmarks",frame_landmarks)
        cv2.imshow("video",frame)


        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break



