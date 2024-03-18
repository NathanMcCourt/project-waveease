import cv2
import mediapipe as mp
import numpy as np

def InitializeKalmanFilter():
    kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)  # Measurement matrix
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)  # State transition matrix
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Process noise
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.0005  # Measurement noise
    kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1  # Error covariance
    return kalman

def main():
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    kalman = InitializeKalmanFilter()

    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Tracking the wrist as a simple proxy for the hand's center
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_x = wrist.x * img.shape[1]
                wrist_y = wrist.y * img.shape[0]

                # Correct the Kalman filter with the detected position and predict the next state
                kalman.correct(np.array([[np.float32(wrist_x)], [np.float32(wrist_y)]]))
                predicted = kalman.predict()

                # Use the Kalman filter's prediction to draw the circle
                cv2.circle(img, (int(predicted[0]), int(predicted[1])), 10, (0, 255, 0), -1)

                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hands", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
