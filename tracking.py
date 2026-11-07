from common import *
import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
  HandLandmarker,
  HandLandmarkerOptions,
  RunningMode,
)
import overlay
import pandas as pd


recognitionModel = joblib.load("models/gesture_model.pkl")
print("loaded model!")

LANDMARKER_PATH = "models/hand_landmarker.task"

def main():
  download_model(LANDMARKER_PATH)
  # camera setup
  cap = cv2.VideoCapture(cv2.CAP_DSHOW)
  print("loaded camera!")

  if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

  # model setup
  # should i try with LIVE_VIEW? 
  # it will be async and no lag,

  header = []
  for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]
  #header.append("label") dumbass mistake lol

  overlay.init()

  options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.5,
  )

  start_time = time.time()
  with HandLandmarker.create_from_options(options) as detector:
    print("loaded landmarker!")
    
    while True:
      ok, frame = cap.read()
      if not ok:
        break

      frame = cv2.flip(frame, 1)
      rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      timestamp_ms = int((time.time() - start_time) * 1000)
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
      results = detector.detect_for_video(mp_image, timestamp_ms)
      prediction = "none"

      if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
          draw_hand(frame, hand_landmarks)
        
          row = []
          for landmark in hand_landmarks:
            row = normalize_static(hand_landmarks) # NORMALISATION !!!!

          #dataframe = pd.DataFrame(row, columns=header) # technically not needed, if i remove it will spam
          prediction = recognitionModel.predict([row]) # HERE BE DRAGONS - the magic happens here


          w,h = overlay.get_screen_size()
          wx = int((w-5)*hand_landmarks[INDEX_TIP].x+2)
          hx = int((h-5)*hand_landmarks[INDEX_TIP].y+2)
          overlay.move_dot(wx,hx)
          overlay.tick()

      cv2.putText(frame, f"GESTURES: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2, cv2.LINE_AA)


      cv2.imshow("Hands Off preview", frame)
      key = cv2.waitKey(1) & 0xFF
      if key == ord("q"):
        break
      if key == ord("s"):
        filename = f"frame_{int(time.time() * 1000)}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()