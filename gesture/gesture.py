import cv2
import numpy as np
from handDetector import HandDetector
import time
import autopy
import win32gui, win32process, psutil
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volumeRange = volume.GetVolumeRange()  # (-63.5, 0.0, 0.03125)
minVol = volumeRange[0]
maxVol = volumeRange[1]

# Create windows
wScr, hScr = autopy.screen.size()  # Returns the width and height of the computer screen (1920.0, 1080.0)
wCam, hCam = 1280, 720  # Width and height of the video display window
pt1, pt2 = (100, 100), (1000, 500)  # The range of movement of the virtual mouse, upper left coordinate pt1, lower right coordinate pt2

cap = cv2.VideoCapture(0)  #
cap.set(3, wCam)  #
cap.set(4, hCam)  #

pTime = 0  # Sets the start time for the first frame to begin processing
pLocx, pLocy = 0, 0  # Mouse position at the previous frame
smooth = 5  # Customize the smoothing factor to smooth out the mouse movement a bit
frame = 0  # Initialize the cumulative number of frames
toggle = False  # flag variable
prev_state = [1, 1, 1, 1, 1]  # Initialize the previous frame state
current_state = [1, 1, 1, 1, 1]  # Initialize the current positive state

# Receive hand detection methods
detector = HandDetector(mode=False,
                        maxHands=1,
                        detectionCon=0.8,
                        minTrackCon=0.5)

# Process frame
while True:

    success, img = cap.read()

    img = cv2.flip(img, flipCode=1)  # 1 for horizontal flip, 0 for vertical flip
    # Create a rectangular box on the image window and move the mouse within the area
    cv2.rectangle(img, pt1, pt2, (0, 255, 255), 5)
    # Determine the process name of the currently active window
    try:
        pid = win32process.GetWindowThreadProcessId(win32gui.GetForegroundWindow())
        print("pid:", pid)
        active_window_process_name = psutil.Process(pid[-1]).name()
        print("acitiveprocess:", active_window_process_name)
    except:
        pass
    # hand landmarker detection
    # Pass in each frame, return the coordinates of the hand keypoints (dictionary), draw the image after the keypoints.
    hands, img = detector.findHands(img, flipType=False, draw=True)
    print("hands:", hands)
    # [{'lmList': [[889, 652, 0], [807, 613, -25], [753, 538, -39], [723, 475, -52], [684, 431, -66], [789, 432, -27],
    #              [762, 347, -56], [744, 295, -78], [727, 248, -95], [841, 426, -39], [835, 326, -65], [828, 260, -89],
    #              [820, 204, -106], [889, 445, -54], [894, 356, -85], [892, 295, -107], [889, 239, -123],
    #              [933, 483, -71], [957, 421, -101], [973, 376, -115], [986, 334, -124]], 'bbox': (684, 204, 302, 448),
    #   'center': (835, 428), 'type': 'Right'}]
    # If the hand can be detected then proceed to the next step
    if hands:

        # Get information about the 21 key points in the hands informationhands
        lmList = hands[0]['lmList']  # The hands are a list of N dictionaries containing information about the key points of each hand, here representing hand 0
        hand_center = hands[0]['center']
        drag_flag = 0
        # Get the coordinates of the tip of the index finger and the tip of the middle finger.
        x1, y1, z1 = lmList[8]  # The index number of the key point at the tip of the index finger is 8
        x2, y2, z2 = lmList[12]  # Middle Finger Index 12

        # Calculate the coordinates of the midpoint
        # between the index and middle fingers.
        cx, cy, cz = (x1 + x2) // 2, (y1 + y2) // 2, (z1 + z2) // 2

        hand_cx, hand_cy = hand_center[0], hand_center[1]
        # Check which finger is facing upwards
        fingers = detector.fingersUp(hands[0])
        print("fingers", fingers)  # Return [0,1,1,0,0] means only the index and middle fingers are up.

        # Calculate the distance between the tip of the index finger and the tip of the middle finger, draw the image
        # img, and the information of the fingertip line info.
        distance, info, img = detector.findDistance((x1, y1), (x2, y2), img)
        # Determine the range of mouse movement
        # Maps the range of movement of the index finger tip from a pre-made window range to the computer screen range.
        x3 = np.interp(x1, (pt1[0], pt2[0]), (0, wScr))
        y3 = np.interp(y1, (pt1[1], pt2[1]), (0, hScr))
        # Center of hand coordinates mapped to screen range
        x4 = np.interp(hand_cx, (pt1[0], pt2[0]), (0, wScr))
        y4 = np.interp(hand_cy, (pt1[1], pt2[1]), (0, hScr))
        # Smooth so that the mouse arrow doesn't keep wiggling when the finger is moving the mouse
        cLocx = pLocx + (x3 - pLocx) / smooth  # Current mouse position coordinates
        cLocy = pLocy + (y3 - pLocy) / smooth
        # Record the current gesture state
        current_state = fingers
        # Record the number of frames in the same state
        if (prev_state == current_state):
            frame = frame + 1
        else:
            frame = 0
        prev_state = current_state
        if fingers != [0, 0, 0, 0, 0] and toggle and frame >= 2:
            autopy.mouse.toggle(None, False)
            toggle = False
            print("Release left hold")
            # With only the index and middle fingers up, it is considered to be moving the mouse
            if fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2 and frame >= 1:
                # move mouse
                autopy.mouse.move(cLocx, cLocy)  # Give the coordinates of the mouse movement position

                print("Move mouse")

                # Update the coordinates of the mouse position of the previous frame.
                pLocx, pLocy = cLocx, cLocy

                # If the index and middle fingers are up and the distance between the fingertips is less than a certain
                # value, it is considered to be a mouse click. A mouse click is considered a mouse click when the distance between the fingers is less than 43 (pixel distance)
                if distance < 43 and frame >= 1:
                    # Draw a green circle on the tip of your index finger to indicate that you are clicking the mouse
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)

                    # left click on the mouse
                    autopy.mouse.click(button=autopy.mouse.Button.LEFT, delay=0)
                    cv2.putText(img, "left_click", (150, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                    print("Left Click")
                else:
                    cv2.putText(img, "move", (150, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            # Bend your middle finger and put your index finger on top. Right click on the mouse.
            elif fingers[1] == 1 and fingers[2] == 0 and sum(fingers) == 1 and frame >= 2:
                autopy.mouse.click(button=autopy.mouse.Button.RIGHT, delay=0)
                print("Right click")
                cv2.putText(img, "rigth_click", (150, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)

            # Five-finger grip, press left button firmly to drag and drop
            elif fingers == [0, 0, 0, 0, 0]:
                if toggle == False:
                    autopy.mouse.toggle(None, True)
                    print("Hold left click")
                toggle = True
                autopy.mouse.move(cLocx, cLocy)
                pLocx, pLocy = cLocx, cLocy
                cv2.putText(img, "drag", (150, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                print("Drag mouse")

            # Thumbs open, others bent, press up button once
            elif fingers == [1, 0, 0, 0, 0] and frame >= 2:
                cv2.putText(img, "UP", (150, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                if (active_window_process_name == "spotify.exe"):
                    print("#############################################")
                    autopy.key.toggle(autopy.key.Code.LEFT_ARROW, True, [autopy.key.Modifier.CONTROL])
                    autopy.key.toggle(autopy.key.Code.LEFT_ARROW, False, [autopy.key.Modifier.CONTROL])
                    print("Last play")
                    time.sleep(0.3)
                else:
                    autopy.key.toggle(autopy.key.Code.UP_ARROW, True, [])
                    autopy.key.toggle(autopy.key.Code.UP_ARROW, False, [])
                    print("Up and down")

                    time.sleep(0.3)

    # Display image
    # FPS Show
    cTime = time.time()  # Time to finish processing a frame
    fps = 1 / (cTime - pTime)
    pTime = cTime  # Reset start time
    print(fps)
    # Display fps information on video, convert to integer and then to string, text display coordinates, text font, text size
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display image, input window name and image data
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == 27:  # Each frame lags for 20 milliseconds and then disappears, ESC key to exit
        break

# Release of video resources
cap.release()
cv2.destroyAllWindows()

# if __name__ == '__main__':