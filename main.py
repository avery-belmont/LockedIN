import argparse
import numpy as np
import cv2
from requests import options
from  ultralytics import YOLO
import supervision as sv
from playsound3 import playsound
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


#initializing mediapipe hand detection and drawing utilities
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE
)
hand_detector = vision.HandLandmarker.create_from_options(options)


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

def play_deterrent():
            audio_thread = threading.Thread(target=playsound, args=("skeleton.mp3",)) # play deterrent sound in separate thread
            audio_thread.start()
            
            deterrent = cv2.VideoCapture("skeleton.mp4") # load deterrent video
            
            while deterrent.isOpened():
                ret, vframe = deterrent.read()
                if not ret:
                    break

                vframe = cv2.resize(vframe, (640, 640)) # resize video frame to match webcam resolution
                cv2.imshow('LOCK IN!!!', vframe)
                if cv2.waitKey(30) == ord('q'): # press q to end deterrent video
                    break
            deterrent.release()
            cv2.destroyWindow('LOCK IN!!!')  # Only close this window, not all
            audio_thread.join() # wait for audio thread to finish before continuing with main loop

def is_hand_holding_phone(hand_landmarks, phone_box, frame_width, frame_height):
    x1, y1, x2, y2 = phone_box # get bounding box coordinates of phone detection
    for landmark in hand_landmarks:
            x = int(landmark.x * frame_width) # convert normalized coordinates to pixel coordinates (assuming 1280x720 resolution)
            y = int(landmark.y * frame_height)
            if x1 <= x <= x2 and y1 <= y <= y2: # check if hand landmark is within phone bounding box
                return True
    return False


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

    
    


    phone_last_frame = False # variable to track if phone was detected in the last frame

    while True:
        ret, frame = cap.read() # returns frame (image numpy array), ret tells you if capture works properly 
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2] # get actual frame dimensions


        #convert frame for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        #detect hands
        hand_results = hand_detector.detect(mp_image)
        
        # DEBUG: Print if hands detected
        print(f"Hands detected: {len(hand_results.hand_landmarks) if hand_results.hand_landmarks else 0}")

        result = model(frame)[0] # get the first result (since we are processing one frame at a time)
        detections = sv.Detections.from_ultralytics(result)

        #draw hand landmarks on frame
        if hand_results.hand_landmarks:
            for hand_landmarks in hand_results.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1]) # convert normalized coordinates to pixel coordinates
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) # draw green circle at each landmark point
                    
        results = model(frame)[0] # run YOLOv8 model on the frame
        detections = sv.Detections.from_ultralytics(results) # convert YOLOv8 results to Supervision Detections format

        mask = detections.class_id == 67
        phone_detections = detections[mask] # filter for phone detections (class_id 67 corresponds to cell phone in COCO dataset)
        
       
        hand_holding_phone = False
        if len(phone_detections) > 0 and hand_results.hand_landmarks: # if phone is detected and hand landmarks are detected
            phone_box = phone_detections[0].xyxy[0] # get bounding box coordinates of first phone detection
            #check each detected hand
            for hand_landmarks in hand_results.hand_landmarks:
                if is_hand_holding_phone(hand_landmarks, phone_box, frame_width, frame_height): # check if hand is holding phone
                    hand_holding_phone = True
                    break

        if len(phone_detections) > 0 and not phone_last_frame and hand_holding_phone: # if phone is detected and it was not detected in the last frame
            play_deterrent()
            phone_last_frame = True # set variable to true to indicate phone was detected in this frame
        elif not hand_holding_phone: # if no hand is holding the phone, reset the variable
            phone_last_frame = False
            #cv2.imshow('Phone Detections', frame)

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=phone_detections)
        #annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=phone_detections)

        cv2.imshow('Phone Detections', annotated_frame)
        
       
       
        
        if cv2.waitKey(1) == ord('q'): # press q to end window
            break

    cap.release()
    cv2.destroyAllWindows()

    


if __name__ == "__main__":
    main()