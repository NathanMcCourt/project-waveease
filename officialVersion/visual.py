import cv2
import mediapipe as mp
import numpy as np

import drawUtiles as dw

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'+format(result.gestures[0][0].category_name))


model_path = 'own_model.task'
base_options = BaseOptions(model_asset_path=model_path)
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
    )
recognizer = GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
previous_position = None
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    gesture_recognition_result = recognizer.recognize_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
    if gesture_recognition_result.gestures:
        print(gesture_recognition_result.gestures[0][0].category_name)


    cv2.imshow('Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
