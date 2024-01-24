import cv2




vid = cv2.VideoCapture('video5.mp4') 
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video5.avi', fourcc, 60.0, (1440,1440))

while True: 
    
  
    ret, original_frame = vid.read() 
    # Check if the frame is read successfully
    

    grey_img = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    invert = cv2.bitwise_not(grey_img)

    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    invertedblur = cv2.bitwise_not(blur)
    
    # VALUES THAT WORK GOOD.
    # 1. 260.0 - 259.0
    # 2. 225
    
    sketch = cv2.divide(grey_img, invertedblur, scale=260.0)
    

    
    
    
    cv2.imshow("Sketch Image",sketch)
    out.write(sketch)
    
      
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
