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