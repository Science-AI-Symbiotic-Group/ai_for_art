import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

# INITIALISING IMPORTANT VARIABLES
model_path = 'D:/VsCodeProjects/Python/Github Projects/video_to_doodle/mediapipe_try/pose_landmarker_heavy.task'
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
# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))
  


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
)


vid = cv2.VideoCapture(0) 

open("results.txt","w").close()
while True:         
    with PoseLandmarker.create_from_options(options) as landmarker:

        ret, original_frame = vid.read() 
        # The landmarker is initialized. Use it here.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=original_frame)
        detection_result = landmarker.detect_for_video(mp_image,frame_timestamp)
        print(type(detection_result))
        frame_timestamp = frame_timestamp + 1
        
        #original_frame = draw_landmarks_on_image(original_frame, detection_result)
        #cv2.imshow("hello",original_frame)
        with open("results.txt","a+") as results_file:
          if str(detection_result) == "PoseLandmarkerResult(pose_landmarks=[], pose_world_landmarks=[], segmentation_masks=None)":
            results_file.writelines(f"Not Detected\n")
          else:
            results_file.writelines(f"{str(detection_result)}\n")
        

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 