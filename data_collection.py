import csv
import os
import cv2
import mediapipe as mp
import time
import urllib.request
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
  HandLandmarker,
  HandLandmarkerOptions,
  RunningMode,
)
from common import *

TRAINING_FILE_PATH = "gestureRecognition/gestures_dataset_"
IMAGES_TO_SAVE = 150
GESTURES = ["closed", "open", "middle_finger","point","point_pulled","pinch","scale"]

LANDMARKER_PATH = "models/hand_landmarker.task" 
INSTRUCTIONS = ["Please press [Enter] to begin.", "Please move your hand around, while keeping gesture. Make sure hand is visible.","Please press [Enter] to end."]
download_model(LANDMARKER_PATH)

# initialize headers
dataset_counter = 1
while (os.path.exists(TRAINING_FILE_PATH+str(dataset_counter)+".csv")):
  dataset_counter+=1
with open(TRAINING_FILE_PATH+str(dataset_counter)+".csv", mode="w", newline="") as train_file:
  writer = csv.writer(train_file)
  header = []
  for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]
  header.append("label")
  writer.writerowheader if hasattr(writer, "writerowheader") else writer.writerow(
      header
  ) 



# camera setup
cap = cv2.VideoCapture(cv2.CAP_DSHOW)
if not cap.isOpened():
  raise RuntimeError("Cannot open camera")

# model setup
# should i try with LIVE_VIEW? 
# it will be async and no lag
options = HandLandmarkerOptions(
  base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
  running_mode=RunningMode.VIDEO,
  num_hands=1,
  min_hand_detection_confidence=0.6,
  min_tracking_confidence=0.5,
)

for gesture_label in GESTURES:
  saved_images=0
  status = 0
  start_time = time.time()
  with open(TRAINING_FILE_PATH+str(dataset_counter)+".csv", mode="a", newline="") as train_file:
    with HandLandmarker.create_from_options(options) as detector:
      saveable_data = []
      while True:
        ok, frame = cap.read()
        if not ok:
          break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp_ms = int((time.time() - start_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect_for_video(mp_image, timestamp_ms)

        if status == 0:
          key = cv2.waitKey(1) & 0xFF
          if (key == ord('\n') or key == ord('\r') ):
            status = 1
            print("ENTER!")

        if results.hand_landmarks:
          for hand_landmarks in results.hand_landmarks:
            draw_hand(frame, hand_landmarks)
            if status == 1:
              row = normalize_static(hand_landmarks) # NORMALISATION !!!!

              if gesture_label is not None:
                row.append(gesture_label)
                saveable_data.append(row)
                saved_images +=1


              key = cv2.waitKey(5) & 0xFF
              if (key == ord('\n') or key == ord('\r') ):
                status = 2 if saved_images >= IMAGES_TO_SAVE else 0
                print("ENTER!")
              if (key == ord('q') or key == ord(' ') ):
                status = 2
                saved_images = 0


        cv2.putText(frame, f"GESTURE: {gesture_label}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, INSTRUCTIONS[status], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{gesture_label}: {saved_images}/{IMAGES_TO_SAVE}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Gesture Data Gathering", frame)
        
        if status == 2:
          break
        #if saved_images >= IMAGES_TO_SAVE:
      #  status = 0
      #  break


    for row in saveable_data:
      writer = csv.writer(train_file)
      writer.writerow(row)
      print(f"saved {gesture_label} data")
      


cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

  