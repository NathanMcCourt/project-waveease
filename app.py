import platform
import subprocess
import tkinter as tk
from oldversion import utile as utile
import gesture.mouse_simulator as ms
import officialVersion.gesture_recognition as gs
from oldversion import camera as ca
from oldversion.cleanup import cleanup
from tkinter import messagebox, ttk

import time
import os

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

hotkey_entry = tk.Entry

def start_mouse_simulation():
    ms.start_recognition()
    messagebox.showinfo("message", "simulation closed!")


def start_gesture_recognition():
    gs.start()
    messagebox.showinfo("message", "recognition closed!")

config = configparser.ConfigParser()

def load_settings():
    try:
        config.read('config.ini')
        value = config.get('hotkey', 'value')
        #hotkey_entry.delete(0, tk.END)
        hotkey_entry.insert(hotkey_entry, 0, value)
    except Exception as e:
        messagebox.showerror('Error', f'fail {str(e)}')

def save_settings(selected_camera, selected_music_app, hotkey):
    # Update the global setting
    settings["selected_camera"] = selected_camera.get()
    settings["selected_music_app"] = selected_music_app.get()
    settings["hotkey"] = hotkey.get()

    config['hotkey'] = {'value': hotkey.get()}
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    messagebox.showinfo('Saved', 'Saved to file')

    messagebox.showinfo("Save Setting", "Configuration saved！")


def open_settings():

    load_settings()
    
    # here to open the setting page
    settings_window = tk.Toplevel(root)
    settings_window.title("Setting")
    settings_window.geometry("300x250")

    # Camera and music app
    avaible_camera = ca.find_available_cameras()
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
    cleanup()
    root.destroy()
    exit(0)


def launch_app():
    spotify_path = utile.find_spotify_path()
    if spotify_path:
        print(f"Launching Spotify from: {spotify_path}")
        if platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", spotify_path])
        else:
            subprocess.Popen([spotify_path])
    else:
        print("Spotify installation not found.")


# Create the main windows
root = tk.Tk()
root.title("WavEase!")
root.geometry("500x400")  # initial size
root.minsize(500, 300)
root.configure(background="#87CEEB")

# layout
root.grid_columnconfigure([1, 2], weight=1, minsize=70)
# root.grid_columnconfigure(1, weight=1, minsize=70)

root.grid_rowconfigure(0, weight=1, minsize=50)
root.grid_rowconfigure(1, weight=1, minsize=10)
#root.grid_rowconfigure([2,3], weight=1, minsize=10) ## does not work for Mac
root.attributes('-alpha', 1.0) 

# Add a label
#label = tk.Label(root, text=r"""\
# __          __         _                                       _              __          __                  ______                        
# \ \        / /        | |                                     | |             \ \        / /                 |  ____|                       
#  \ \  /\  / /    ___  | |   ___    ___    _ __ ___     ___    | |_    ___      \ \  /\  / /    __ _  __   __ | |__      __ _   ___    ___   
#   \ \/  \/ /    / _ \ | |  / __|  / _ \  | '_ ` _ \   / _ \   | __|  / _ \      \ \/  \/ /    / _` | \ \ / / |  __|    / _` | / __|  / _ \  
#    \  /\  /    |  __/ | | | (__  | (_) | | | | | | | |  __/   | |_  | (_) |      \  /\  /    | (_| |  \ V /  | |____  | (_| | \__ \ |  __/  
#     \/  \/      \___| |_|  \___|  \___/  |_| |_| |_|  \___|    \__|  \___/        \/  \/      \__,_|   \_/   |______|  \__,_| |___/  \___|""", background='#c3c3c3')
label = tk.Label(root, text= r""" ༄ Welcome to WavEase ༄ """, background='#87CEEB', fg="white")
label.configure(font = ("Comic Sans MS", 28, "bold"))
label.grid(row=0, column=1, sticky="nsew", padx=20, pady=10, columnspan=2)

#tree graphic
frameCnt = 120
frames = [tk.PhotoImage(file='.assets/tree-01.gif',format = 'gif -index %i' %(i)) for i in range(frameCnt)]

def update(ind):

    frame = frames[ind]
    ind += 1
    if ind == frameCnt:
        ind = 0
    tree.configure(image=frame)
    root.after(100, update, ind)
tree = tk.Label(root, background='#87CEEB')
tree.grid(row=1, column=1, sticky="nsew", columnspan=2)
root.after(0, update, 0)

# Add Button
start_button = tk.Button(root, text="Start Recognition", command=start_gesture_recognition)
start_button.grid(row=2, column=1, padx=8, pady=8, ipadx=30, ipady=5)

start_button = tk.Button(root, text="Start Mouse Simulation", command=start_mouse_simulation)
start_button.grid(row=2, column=2, padx=8, pady=8, ipadx=30, ipady=5)

launch_button = tk.Button(root, text="Launch app", command=launch_app)
launch_button.grid(row=2, column=3, padx=8, pady=8, ipadx=30, ipady=5)

settings_button = tk.Button(root, text="Settings", command=open_settings)
settings_button.grid(row=3, column=1, padx=8, pady=8, ipadx=30, ipady=5)

exit_button = tk.Button(root, text="Exit", command=exit_app)
exit_button.grid(row=3, column=2, padx=8, pady=8, ipadx=30, ipady=5)

# start the evert loop
root.mainloop()
