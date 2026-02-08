import argparse
import numpy as np
import cv2
from  ultralytics import YOLO
import supervision as sv


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
    #frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    
    #set width and height for larger frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # height

    model = YOLO("yolov8s.pt")

    box_annotator = sv.BoxAnnotator(
        thickness = 2,
    )

    label_annotator = sv.LabelAnnotator(
        text_thickness = 2,
        text_scale = 1
    )

    
    


    

    while True:
        ret, frame = cap.read() # returns frame (image numpy array), ret tells you if capture works properly 
        if not ret:
            break
        
        result = model(frame)[0] # get the first result (since we are processing one frame at a time)
        detections = sv.Detections.from_ultralytics(result)

        #print(f"Detected classes: {detections.class_id}")
        #print(f"Confidences: {detections.confidence}")

        mask = detections.class_id == 67
        phone_detections = detections[mask] # filter for phone detections (class_id 67 corresponds to cell phone in COCO dataset)
        
        if len(phone_detections) > 0:
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=phone_detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=phone_detections)
            cv2.imshow('Phone Detections', annotated_frame)
        else:
            cv2.imshow('Phone Detections', frame)

        
        #annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=phone_detections)
       
       
        
        if cv2.waitKey(1) == ord('q'): # press q to end window
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()