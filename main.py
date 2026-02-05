import argparse
import numpy as np
import cv2
from  ultralytics import YOLO


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument(
         "--webcam-resolution",
           nargs = "+", 
           type = int, 
           default = [1280, 720]
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    
    #set width and height for larger frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width) # width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height) # height

    while True:
        ret, frame = cap.read() # returns frame (image numpy array), ret tells you if capture works properly 
    
        cv2.imshow('frame', frame)

       
        
        if cv2.waitKey(1) == ord('q'): # press q to end window
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()