import cv2
import mediapipe as mp
import numpy as np

class LandmarkKalmanFilter:
    """Class to encapsulate Kalman filter setup for smoothing landmark movements."""
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)  # Measurement matrix
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)  # State transition matrix
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.35  # Process noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.005  # Measurement noise
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1  # Error covariance

    def predict(self):
        """Predict the next state."""
        return self.kalman.predict()

    def correct(self, measurement):
        """Correct the state with the latest measurement."""
        return self.kalman.correct(measurement)


def main():
    """Main function to detect hand gestures using MediaPipe and smooth landmarks using Kalman filter."""
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    kalman_filters = [LandmarkKalmanFilter() for _ in range(21)]  # Initialize a Kalman filter for each landmark

    previous_position = None  # Store the previous wrist position

    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_position = np.array([wrist_landmark.x * img.shape[1], wrist_landmark.y * img.shape[0]])

                if previous_position is not None:
                    # Calculate movement direction
                    movement = wrist_position - previous_position
                    if abs(movement[0]) > abs(movement[1]):  # Horizontal movement
                        if movement[0] > 0:
                            direction = "Right"
                        else:
                            direction = "Left"
                    else:  # Vertical movement
                        if movement[1] > 0:
                            direction = "Down"
                        else:
                            direction = "Up"

                    print(f"Gesture moved: {direction}")

                previous_position = wrist_position

                for i, landmark in enumerate(hand_landmarks.landmark):
                    # Update Kalman filter for each landmark
                    kalman_filter = kalman_filters[i]
                    measurement = np.array([[np.float32(landmark.x * img.shape[1])], [np.float32(landmark.y * img.shape[0])]])
                    kalman_filter.correct(measurement)
                    predicted = kalman_filter.predict()

                    # Draw circles at the predicted positions for all landmarks
                    cv2.circle(img, (int(predicted[0]), int(predicted[1])), 5, (0, 255, 0), -1)

                    # get the keypoints coordination
                    landmarks = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]
                    # calculate the coordination from landmarks
                    min_x = min([coord[0] for coord in landmarks])
                    max_x = max([coord[0] for coord in landmarks])
                    min_y = min([coord[1] for coord in landmarks])
                    max_y = max([coord[1] for coord in landmarks])

                    # Convert coordinates from relative values to actual pixel coordinates
                    min_x, max_x = int(min_x * img.shape[1]), int(max_x * img.shape[1])
                    min_y, max_y = int(min_y * img.shape[0]), int(max_y * img.shape[0])

                    # Draw bounding box on image
                    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

                # Draw MediaPipe hand landmarks
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        cv2.imshow("Hands", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()