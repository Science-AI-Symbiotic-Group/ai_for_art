import cv2
import numpy as np
import argparse
import os

argument_parser = argparse.ArgumentParser()

argument_parser.add_argument("video")
argument_parser.add_argument("--background_video",help="Select the Video you want for the Background. Must be a Video File")
argument_parser.add_argument("--background_gamma",help="The Value for the Gamma of the Background Video, By Default is 1.0")


arguments = argument_parser.parse_args()

using_bg = True

if arguments.background_gamma == None:
    background_gamma = 1.0  #default value for gamma
else:
    background_gamma = float(arguments.background_gamma)




if arguments.background_video == None:
    using_bg = False
    print("Not Using a Background Video")
else:
    if os.path.exists(arguments.background_video) == False:
        print(f"The Background Video does not exist")
        raise SystemExit(1)
    else:
        using_bg = True


if os.path.exists(arguments.video) == False:
    print("The video does not exist")
    raise SystemExit(1)





cap = cv2.VideoCapture(arguments.video)   #replace with webcam number or file name

if using_bg == True:
    cap_bg = cv2.VideoCapture(arguments.background_video)   #replace with background file name

# Get the video details (width, height, frames per second)
width = cap.get(3)
height = cap.get(4)
fps = cap.get(5)

print(f"fps: {fps}")
print(f"width: {width}")
print(f"height:  {height}")

width_int = int(width)
height_int = int(height)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width_int, height_int))

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)




while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if using_bg == True:
        ret_bg, frame_bg = cap_bg.read()
    

    # Check if the frame is read successfully
    if not ret:
        print("Error: Couldn't read a frame.")
        break


    try:
        if using_bg == True:
            frame_bg = cv2.resize(frame_bg, (width_int,height_int))        
            frame_bg = adjust_gamma(frame_bg,1.0)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray=cv2.medianBlur(gray,1)
        gray = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=100) 
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
        cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 1)
        
        if using_bg == True:
            contour_frame = cv2.add(contour_frame,frame_bg)
        
        out.write(contour_frame)


        

        # Display the frame with contours on a dark background
        cv2.imshow("Contours on Dark Background", contour_frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    except:        
        print("Looping Background Video.")
        cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_bg, frame_bg = cap_bg.read()

    


cap.release()
cv2.destroyAllWindows()
