import cv2
import mediapipe as mp
import numpy as np


class LandmarkKalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)  # Measurement matrix
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)  # State transition matrix
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Process noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.0005  # Measurement noise
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1  # Error covariance

    def predict(self):
        return self.kalman.predict()

    def correct(self, measurement):
        return self.kalman.correct(measurement)


def main():
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    kalman_filter = LandmarkKalmanFilter()

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
                wrist_x = wrist_landmark.x * img.shape[1]
                wrist_y = wrist_landmark.y * img.shape[0]

                # Update Kalman filter with detected wrist position and get the predicted position
                measurement = np.array([[np.float32(wrist_x)], [np.float32(wrist_y)]])
                kalman_filter.correct(measurement)
                predicted = kalman_filter.predict()

                # Draw the predicted wrist position
                cv2.circle(img, (int(predicted[0]), int(predicted[1])), 10, (0, 255, 0), -1)

                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hands", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
