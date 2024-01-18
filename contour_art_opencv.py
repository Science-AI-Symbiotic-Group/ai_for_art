import cv2
import numpy as np

cap = cv2.VideoCapture('videos/6.mp4')   #replace with webcam number or file name
cap_bg = cv2.VideoCapture('background/bg_1.mp4')   #replace with background file name
#frame_bg = cv2.imread('background/2.jpg')   #replace with background file name for image

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
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video1.avi', fourcc, fps, (width_int, height_int))

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)




while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    ret_bg, frame_bg = cap_bg.read()
    

    # Check if the frame is read successfully
    if not ret:
        print("Error: Couldn't read a frame.")
        break


    try:
        frame_bg = cv2.resize(frame_bg, (width_int,height_int))
        

        frame_bg = adjust_gamma(frame_bg,1.0)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray=cv2.medianBlur(gray,1)
        gray = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=10) 
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
