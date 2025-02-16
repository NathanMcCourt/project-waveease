import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


TIMEOUT_SECONDS = 5

# Initial ImageBind model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)


def analyze_with_imagebind(file_path):
    # List of articles by category
    inputs = None
    lables = ["fist", "thumbs", "slap", "fourth", ""]

    if file_path.endswith('.avi'):
        inputs = {
            ModalityType.TEXT:data.load_and_transform_text(lables,device),
            ModalityType.VISION: data.load_and_transform_video_data([file_path], device),
        }
    elif file_path.endswith('.jpg'):
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(lables, device),
            ModalityType.VISION: data.load_and_transform_vision_data([file_path], device),
        }

    if inputs is not None:
        with torch.no_grad():
            embeddings = model(inputs)
        scores = (
            torch.softmax(
                embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim = -1
            )
            .squeeze(0)
            .tolist()
        )
        score_dict = {label: score for label, score in zip(lables, scores)}
        max_label = max(score_dict, key=score_dict.get)  # Finds the label with the highest score
        print("Analysis completed：", score_dict)
        print(f"Analysis completed： {max_label} with a probability of {score_dict[max_label]:.2f}")

    else:
        print("Unsupportive sentence category")


class LandmarkKalmanFilter:
    """Class to encapsulate Kalman filter setup for smoothing landmark movements."""

    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)  # Measurement matrix
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                np.float32)  # State transition matrix
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.35  # Process noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.005  # Measurement noise
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1  # Error covariance

    def predict(self):
        """Predict the next state."""
        return self.kalman.predict()

    def correct(self, measurement):
        """Correct the state with the latest measurement."""
        return self.kalman.correct(measurement)


class MovementDetector:
    def __init__(self, window_size=2.0, move_threshold=10):
        self.window_size = window_size
        self.move_threshold = move_threshold
        self.previous_positions = []
        self.window_start_time = time.time()

    def update_position(self, position):
        current_time = time.time()
        self.previous_positions.append((current_time, position))
        self.previous_positions = [pos for pos in self.previous_positions if current_time - pos[0] <= self.window_size]

    def has_moved(self):
        if len(self.previous_positions) > 1:
            start_position = self.previous_positions[0][1]
            end_position = self.previous_positions[-1][1]
            displacement = np.linalg.norm(end_position - start_position)
            return displacement >= self.move_threshold
        return False


def find_available_cameras(max_tests=10):
    available_cameras = []
    for i in range(max_tests):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
        else:
            break
    return available_cameras


available_cameras = find_available_cameras()
print("Available Camera devices：", available_cameras)

if not os.path.exists('captures'):
    os.makedirs('captures')
    print("Created captures directory")
else:
    print("captures directory already exists")
if not os.path.exists('captures/photos'):
    os.makedirs('captures/photos')
    print("Created photos directory")
else:
    print("photos directory already exists")
if not os.path.exists('captures/videos'):
    os.makedirs('captures/videos')
    print("Created videos directory")
else:
    print("videos directory already exists")


def start_capture():
    """Main function to detect hand gestures using MediaPipe and smooth landmarks using Kalman filter."""
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    frames = []

    significant_movement_detected = False
    is_tracking_time = False
    start_time = time.time()
    last_hand_detected_time = time.time()


    # MediaPipe hands setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    kalman_filters = {}
    movement_detectors = {}
    previous_positions = {}  # Store the previous wrist position

    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            if not is_tracking_time:
                start_time = time.time()  # Start the timer
                is_tracking_time = True
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Check and create Kalman filters for the detected hand
                if hand_index not in kalman_filters:
                    kalman_filters[hand_index] = [LandmarkKalmanFilter() for _ in range(21)]
                    movement_detectors[hand_index] = MovementDetector(window_size=0.5, move_threshold=10)
                    previous_positions[hand_index] = None

                # Kalman filter bounding box
                min_x, min_y = float('inf'), float('inf')
                max_x, max_y = 0, 0

                for i, landmark in enumerate(hand_landmarks.landmark):
                    kalman_filter = kalman_filters[hand_index][i]
                    measurement = np.array(
                        [[np.float32(landmark.x * img.shape[1])], [np.float32(landmark.y * img.shape[0])]])
                    predicted = kalman_filter.correct(measurement)

                    min_x = min(min_x, predicted[0])
                    max_x = max(max_x, predicted[0])
                    min_y = min(min_y, predicted[1])
                    max_y = max(max_y, predicted[1])

                    cv2.circle(img, (int(predicted[0]), int(predicted[1])), 5, (0, 255, 0), -1)

                cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)

                for i, landmark in enumerate(hand_landmarks.landmark):
                    kalman_filter = kalman_filters[hand_index][i]
                    measurement = np.array(
                        [[np.float32(landmark.x * img.shape[1])], [np.float32(landmark.y * img.shape[0])]])
                    kalman_filter.correct(measurement)
                    predicted = kalman_filter.predict()
                    cv2.circle(img, (int(predicted[0]), int(predicted[1])), 5, (0, 255, 0), -1)

                wrist_position = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * img.shape[1],
                                           hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * img.shape[0]])

                movement_detector = movement_detectors[hand_index]
                movement_detector.update_position(wrist_position)

                if movement_detector.has_moved():
                    current_position = wrist_position
                    significant_movement_detected = True
                    if previous_positions[hand_index] is not None:
                        movement = current_position - previous_positions[hand_index]
                        # Determine movement direction
                        if np.linalg.norm(movement) > 1:  # Threshold check
                            horizontal_movement = movement[0]
                            vertical_movement = movement[1]
                            if abs(horizontal_movement) > abs(vertical_movement):
                                direction = "Right" if horizontal_movement > 0 else "Left"
                            else:
                                direction = "Up" if vertical_movement < 0 else "Down"  # Note: screen coordinates y-axis is inverted
                            print(f"Hand {hand_index} moved: {direction}")
                    previous_positions[hand_index] = current_position
                else:
                    print(f"Hand {hand_index}: Minimal or no movement.")

                # Draw MediaPipe hand landmarks
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            frames.append(img)
            # Remove trackers for hands that are no longer detected
            active_hands = set(range(len(results.multi_hand_landmarks)))
            inactive_hands = set(kalman_filters.keys()) - active_hands
            for hand_index in inactive_hands:
                del kalman_filters[hand_index]
                del movement_detectors[hand_index]
                del previous_positions[hand_index]

        else:
            recorded_time = time.time() - last_hand_detected_time
            if recorded_time > TIMEOUT_SECONDS:
                print("No hand detected.")
                print("Pass " + str(TIMEOUT_SECONDS) + " seconds without hand.")
                last_hand_detected_time = time.time()

        # Check if x seconds have passed
        if is_tracking_time:
            if time.time() - start_time > TIMEOUT_SECONDS:
                print("Pass " + str(TIMEOUT_SECONDS) + " seconds.")
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                if significant_movement_detected and frames:
                    video_filename = f'captures/videos/{timestamp}.avi'
                    out = cv2.VideoWriter(video_filename, fourcc, 20.0, frame_size)
                    for frame in frames: out.write(frame)
                    out.release()
                    print(f"Saved video: {timestamp}.avi")
                    analyze_with_imagebind(video_filename)
                    print("Analyzed with imagebind (video)")
                elif frames:
                    photo_filename = f'captures/photos/{timestamp}.jpg'
                    cv2.imwrite(photo_filename, frames[-1])
                    print(f"Saved photo: {timestamp}.jpg")
                    analyze_with_imagebind(photo_filename)
                    print("Analyzed with imagebind (picture)")
                else:
                    print("No frames captured")

                # Reset for the next x seconds
                frames = []
                significant_movement_detected = False
                is_tracking_time = False
                start_time = time.time()
                last_hand_detected_time = time.time()

        cv2.imshow("Hands", img)

        # Use ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
