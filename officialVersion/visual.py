import cv2
import mediapipe as mp
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
    print('gesture recognition result: {}'.format(result))
    annotated_image = dw.draw_landmarks_on_image(frame, result)
    cv2.imshow('Camera Feed', annotated_image)

model_path = 'gesture_recognizer.task'
base_options = BaseOptions(model_asset_path=model_path)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        recognizer.recognize_async(mp_frame, int(cap.get(cv2.CAP_PROP_POS_MSEC)))



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
