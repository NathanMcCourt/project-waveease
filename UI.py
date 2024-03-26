import tkinter as tk
import cv2 as cv

import Camera
import Camera as camera
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

# window = tk.Tk()
# window.title("WAVEASE")
# window.minsize(500, 300)
#
# greeting = tk.Label(text="WELCOME TO WAVEASE")
# greeting.pack()
#
# cam = cv.VideoCapture(0)
# #cam.set(cv.CAP_PROP_FRAME_WIDTH, 200)
# #cam.set(cv.CAP_PROP_FRAME_HEIGHT, 200)
#
# label_camera = tk.Label(window)
# label_camera.pack()
# _, frame = cam.read()
#
#
# def capture():
#     # Capture the video frame by frame
#
#
#
#     window = tk.Tk()
#     window.title("WAVEASE")
#     window.minsize(500, 300)
#
#     greeting = tk.Label(text="WELCOME TO WAVEASE")
#     greeting.pack()
#
#     cam = cv.VideoCapture(0)
#     # cam.set(cv.CAP_PROP_FRAME_WIDTH, 200)
#     # cam.set(cv.CAP_PROP_FRAME_HEIGHT, 200)
#
#     label_camera = tk.Label(window)
#     label_camera.pack()
#         # Convert image from one color space to other
#     opencv_image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
#
#         # Capture the latest frame and transform to image
#     captured_image = Image.fromarray(opencv_image)
#
#         # Convert captured image to photoimage
#     photo_image = ImageTk.PhotoImage(image=captured_image)
#
#         # Displaying photoimage in the label
#     label_camera.photo_image = photo_image
#
#         # Configure image in the label
#     label_camera.configure(image=photo_image)
#
#         # Repeat the same process after every 10 seconds
#     label_camera.after(10, capture)

# capture()
#
# window.mainloop() #listen for events
#
# capture()

settings = {
    "selected_camera": "",
    "selected_music_app": "",
    "hotkey": ""
}

def start_gesture_recognition():
    camera.StartCapture()
    messagebox.showinfo("message", "recognition closed!")


def save_settings(selected_camera, selected_music_app, hotkey):
    # Update the global setting
    settings["selected_camera"] = selected_camera.get()
    settings["selected_music_app"] = selected_music_app.get()
    settings["hotkey"] = hotkey.get()
    messagebox.showinfo("Save Setting", "Configuration saved！")
def open_settings():
    # here to open the setting page
    settings_window = tk.Toplevel(root)
    settings_window.title("Setting")
    settings_window.geometry("300x250")

    # Camera and music app
    avaible_camera = Camera.find_available_cameras()
    camera_options = avaible_camera
    music_app_options = ["App music 1", "Spotify 2"]

    selected_camera = tk.StringVar()
    selected_music_app = tk.StringVar()
    hotkey_entry_var = tk.StringVar()

    # Rollback to previous saved setting
    selected_camera.set(camera_options[0] if camera_options else "No available Camera ")
    selected_music_app.set(settings["selected_music_app"] if settings["selected_music_app"] else music_app_options[0])
    hotkey_entry_var.set(settings["hotkey"])

    # Create UI
    tk.Label(settings_window, text="Select Camera:").grid(row=0, column=0, pady=10, padx=10, sticky="w")
    camera_menu = ttk.Combobox(settings_window, textvariable=selected_camera, values=camera_options)
    camera_menu.grid(row=0, column=1, padx=10, sticky="ew")

    tk.Label(settings_window, text="Select music app:").grid(row=1, column=0, pady=10, padx=10, sticky="w")
    music_app_menu = ttk.Combobox(settings_window, textvariable=selected_music_app, values=music_app_options)
    music_app_menu.grid(row=1, column=1, padx=10, sticky="ew")

    tk.Label(settings_window, text="Set hotkey:").grid(row=2, column=0, pady=10, padx=10, sticky="w")
    hotkey_entry = tk.Entry(settings_window, textvariable=hotkey_entry_var)
    hotkey_entry.grid(row=2, column=1, padx=10, sticky="ew")

    # Create save and back button
    save_button = tk.Button(settings_window, text="Save",
                            command=lambda: save_settings(selected_camera, selected_music_app, hotkey_entry_var))
    save_button.grid(row=3, column=0, pady=10, padx=10, sticky="ew")

    back_button = tk.Button(settings_window, text="Back", command=settings_window.destroy)
    back_button.grid(row=3, column=1, pady=10, padx=10, sticky="ew")

    settings_window.grid_columnconfigure(1, weight=1)


def exit_app():
    root.destroy()


# Create the main windows
root = tk.Tk()
root.title("WavEase!")
root.geometry("800x600")  # initial size

# layout
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)

# Add a lable
label = tk.Label(root, text="Welcome to use Waveease!")
label.grid(row=0, column=0, sticky="nsew")

# Add Button
start_button = tk.Button(root, text="Start Recognition", command=start_gesture_recognition)
start_button.grid(row=1, column=0, sticky="nsew")

settings_button = tk.Button(root, text="Setting", command=open_settings)
settings_button.grid(row=2, column=0, sticky="nsew")

exit_button = tk.Button(root, text="Exit", command=exit_app)
exit_button.grid(row=3, column=0, sticky="nsew")

# start the evert loop
root.mainloop()