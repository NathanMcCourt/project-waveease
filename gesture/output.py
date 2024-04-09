import cv2
import mediapipe as mp
import pyautogui
import numpy as np

def draw_info_text(image, gesture, brect):
    info_text = gesture
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = []
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array.append([landmark_x, landmark_y])
    x, y, w, h = cv2.boundingRect(np.array(landmark_array))
    return [x, y, x + w, y + h]

def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers = []
                for i, finger in enumerate([mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                                            mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]):
                    finger_mcp = hand_landmarks.landmark[finger]
                    finger_tip = hand_landmarks.landmark[finger + 3]
                    fingers.append(finger_tip.y < finger_mcp.y)

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                fingers.insert(0, thumb_tip.x < thumb_ip.x)

                rect = calc_bounding_rect(img, hand_landmarks)

                if fingers[1] and all(not f for f in fingers[2:]):
                    pyautogui.press('volumeup')
                    print("Volume Up")
                    img = draw_info_text(img, "Volume Up", rect)
                elif not fingers[1] and all(not f for f in fingers[2:]):
                    pyautogui.press('volumedown')
                    print("Volume Down")
                    img = draw_info_text(img, "Volume Down", rect)

                if fingers[1] and fingers[4]:
                    # Simulate switching between speakers and headphones
                    #pyautogui.hotkey('ctrl', 'shift', 's')
                    print("Toggle Microphone Output")

                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hands", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()