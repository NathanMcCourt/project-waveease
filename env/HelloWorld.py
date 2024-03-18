
import cv2
import mediapipe as mp
import numpy as np
import time

def main():
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    # Kalman filter implementation (not sure)
    '''
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    measurement = np.array((2, 1), np.float32)
    measurement = np.zeros((2, 1), np.float32)
    '''

    #尝试往右移动打出右，记录基础坐标以判断
    prev_x = None

    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # List used to store finger extension status (True means extended)
                fingers = []

                # Kalman filter implementation (not sure)

                '''
                # wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                # wrist_x, wrist_y = wrist.x, wrist.y
                # predicted = kalman.predict()
                # measurement[0] = wrist_x
                # measurement[1] = wrist_y
                # kalman.correct(measurement)
                # predicted_x, predicted_y = predicted[0], predicted[1]
                # Obtain landmark coordinates for finger MCP (Metacarpophalangeal, metacarpophalangeal joint) and TIP (tip of the finger)
                 '''

                for i, finger in enumerate([mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                                            mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]):
                    finger_mcp = hand_landmarks.landmark[finger]
                    finger_tip = hand_landmarks.landmark[finger + 3]

                    # Determine if fingers are extended (tip y-coordinate is less than metacarpophalangeal joint y-coordinate)
                    fingers.append(finger_tip.y < finger_mcp.y)

                # The thumb is a slightly special case, judged here by its x-coordinate #
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                fingers.insert(0, thumb_tip.x < thumb_ip.x)
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
                # Print the corresponding number based on the number of fingers stretched out

                ##右
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_x, wrist_y = wrist.x, wrist.y

                # Convert coordinates from relative values to actual pixel coordinates
                wrist_x_pixel = int(wrist_x * img.shape[1])

                if prev_x is not None and wrist_x_pixel - prev_x > 20:  # suppost that move right 20 pixels is significant
                    print("left")

                # update the previous position
                prev_x = wrist_x_pixel

                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if all(not f for f in fingers):
                    print("6 - Fist")
                else:
                    print(f"{fingers.count(True)} - Fingers extended")

                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hands", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
