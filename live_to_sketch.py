import cv2

def convert_to_doodle(original_image):
    grey_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(grey_img)

    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    invertedblur = cv2.bitwise_not(blur)
    
    # VALUES THAT WORK GOOD.
    # 1. 260.0 - 259.0
    # 2. 225
    
    sketch = cv2.divide(grey_img, invertedblur, scale=260.0) # 256.0 DEFAULT VALUE
    return sketch


vid = cv2.VideoCapture(0) 
  
while True: 
    width  = vid.get(3)  # float `width`
    height = vid.get(4)  # float `height`
    print(f"width = {width}")
    print(f"height = {height}")
  
    ret, original_frame = vid.read() 
    sketch_image = convert_to_doodle(original_image=original_frame)
    
    cv2.imshow("Sketch Image",sketch_image)
      
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 