import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # returns frame (image numpy array), ret tells you if capture works properly 
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'): # press q to end window
        break

cap.release()
cv2.destroyAllWindows()