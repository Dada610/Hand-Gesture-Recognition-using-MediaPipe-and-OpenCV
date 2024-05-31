# Hand Gesture Recognition using MediaPipe and OpenCV

This project utilizes MediaPipe and OpenCV to perform real-time hand gesture recognition using a webcam. The application captures the live feed from the webcam, detects hand landmarks, and identifies the number of fingers raised. Based on the finger count, it displays custom messages on the screen.

## Features

- Real-time hand detection and finger counting.
- Custom messages displayed based on the number of fingers raised.
- Utilizes MediaPipe for hand landmark detection.
- OpenCV is used for video capture and displaying the results.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Dada610/Hand-Gesture-Recognition-using-MediaPipe-and-OpenCV.git
   cd hand-gesture-recognition
Install the required packages:

bash
pip install opencv-python mediapipe
Usage
Run the script:

bash

python hand_gesture_recognition.py
The script will open a window displaying the webcam feed. It will detect hand landmarks and count the number of fingers raised. Based on the finger count, it will display custom messages.

Code Explanation
The core of this project is a Python script that utilizes OpenCV and MediaPipe to capture video from the webcam, process each frame to detect hand landmarks, and count the number of fingers raised. Here's a breakdown of the main components:

Imports: Import necessary libraries.

python
import cv2
import mediapipe as mp
Initialization: Initialize MediaPipe and OpenCV.

python
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
Hand Detection and Finger Counting: Capture each frame from the webcam, detect hand landmarks, and count the number of fingers raised.

python
Copy code
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    fingerCount = 0

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
        handLabel = results.multi_handedness[handIndex].classification[0].label
        handLandmarks = []
        for landmarks in hand_landmarks.landmark:
          handLandmarks.append([landmarks.x, landmarks.y])
        if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
          fingerCount = fingerCount+1
        elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
          fingerCount = fingerCount+1
        if handLandmarks[8][1] < handLandmarks[6][1]:
          fingerCount = fingerCount+1
        if handLandmarks[12][1] < handLandmarks[10][1]:
          fingerCount = fingerCount+1
        if handLandmarks[16][1] < handLandmarks[14][1]:
          fingerCount = fingerCount+1
        if handLandmarks[20][1] < handLandmarks[18][1]:
          fingerCount = fingerCount+1
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    cv2.putText(image, str(fingerCount), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
    if fingerCount == 1:
      cv2.putText(image, "Hi", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if fingerCount == 2:
      cv2.putText(image, "My name is mortada", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if fingerCount == 3:
      cv2.putText(image, "I am 21 years old", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if fingerCount == 4:
      cv2.putText(image, "I am a student in AIU university ", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if fingerCount == 5:
      cv2.putText(image, "I love computer science ", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
Customization
You can customize the messages displayed based on the finger count by modifying the following section of the code:

python
cv2.putText(image, str(fingerCount), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
if fingerCount == 1:
  cv2.putText(image, "Hi", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
if fingerCount == 2:
  cv2.putText(image, "My name is mortada", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
if fingerCount == 3:
  cv2.putText(image, "I am 21 years old", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
if fingerCount == 4:
  cv2.putText(image, "I am a student in AIU university ", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
if fingerCount == 5:
  cv2.putText(image, "I love computer science ", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

Acknowledgments
MediaPipe by Google
OpenCV
