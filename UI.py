import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk

window = tk.Tk()
window.title("WAVEASE")
window.minsize(500, 300)

greeting = tk.Label(text="WELCOME TO WAVEASE")
greeting.pack()

cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 200)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 200)

label_camera = tk.Label(window)
label_camera.pack()



window.mainloop() #listen for events

# Capture the video frame by frame 
_, frame = cam.read() 
  
    # Convert image from one color space to other 
opencv_image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA) 
  
    # Capture the latest frame and transform to image 
captured_image = Image.fromarray(opencv_image) 
  
    # Convert captured image to photoimage 
photo_image = ImageTk.PhotoImage(image=captured_image) 
  
    # Displaying photoimage in the label 
label_camera.photo_image = photo_image 
  
    # Configure image in the label 
label_camera.configure(image=photo_image) 
  
    # Repeat the same process after every 10 seconds 
label_camera.after(10, open_camera) 