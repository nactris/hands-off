
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
import urllib.request

#///////////////////////////////////////////////////////// CONSTS //////////////////////////////////////////////////////////////////

# Landmark indices (MediaPipe hand topology, 0–20)

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

FINGER_LANDMARK_GROUPS = [
  [WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
  [WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
  [WRIST, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
  [WRIST, RING_MCP, RING_PIP, RING_DIP, RING_TIP],
  [WRIST, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
]

FINGER_COLORS = [
  (0, 140, 255),   # Thumb  — orange
  (0, 220, 0),     # Index  — green
  (220, 0, 220),   # Middle — magenta
  (0, 220, 220),   # Ring   — cyan
  (220, 220, 0),   # Pinky  — yellow
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
# look here https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker#models


#///////////////////////////////////////////////////////// FUNCS //////////////////////////////////////////////////////////////////



def download_model(path):
  if not os.path.exists(path):
    print("Downloading hand_landmarker.task model")
    urllib.request.urlretrieve(MODEL_URL, path)

def landmark_px(landmark, w, h):
  return int(landmark.x * w), int(landmark.y * h)

def draw_hand(frame, hand_landmarks):
  h, w = frame.shape[:2]
  lm = hand_landmarks

  for finger_idx, chain in enumerate(FINGER_LANDMARK_GROUPS):
    color = FINGER_COLORS[finger_idx]
    pts = [landmark_px(lm[i], w, h) for i in chain]
    for a, b in zip(pts, pts[1:]):
      cv2.line(frame, a, b, color, 3, cv2.LINE_AA)
    for pt in pts[1:]:  # skip wrist repetition
      cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)

  # knuckle-line
  for i, j in [(INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP)]:
    cv2.line(frame, landmark_px(lm[i], w, h), landmark_px(lm[j], w, h), (255, 255, 255), 2, cv2.LINE_AA)

  # wrist
  cv2.circle(frame, landmark_px(lm[WRIST], w, h), 7, (255, 255, 255), -1, cv2.LINE_AA)

  # fingertips
  for tip_id, color in zip(FINGER_TIPS, FINGER_COLORS):
    cv2.circle(frame, landmark_px(lm[tip_id], w, h), 9, color, 2, cv2.LINE_AA)


def normalize_static(hand_landmarks):

  pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
  wrist = pts[WRIST]

  pts_centered = pts - wrist
  hand_size = np.linalg.norm(pts_centered[INDEX_MCP])

  if hand_size > 0:
    pts_normalized = pts_centered / hand_size
  else:
    pts_normalized = pts_centered
  
  return pts_normalized.flatten().tolist()



