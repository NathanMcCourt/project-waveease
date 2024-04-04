import os
import subprocess
import platform

import win32process
import win32gui
import psutil
def find_spotify_path():
    os_type = platform.system()
    if os_type == "Windows":
        # the default path for spotify
        path = os.path.expanduser("~\\AppData\\Roaming\\Spotify\\Spotify.exe")
        if os.path.exists(path):
            return path
        # if not exit,try looking for the register to find the path
        try:
            reg_path = r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Spotify"
            reg_query = subprocess.check_output(['reg', 'query', 'HKCU\\' + reg_path, '/v', 'InstallLocation'], encoding='utf-16')
            for line in reg_query.splitlines():
                if "InstallLocation" in line:
                    return line.split("REG_SZ")[1].strip()
        except Exception as e:
            print(e)
            return "Spotify installation path not found on Windows."
    elif os_type == "Darwin":  # macOS
        path = "/Applications/Spotify.app"
        if os.path.exists(path):
            return path
        else:
            return "Spotify installation path not found on macOS."
    else:
        return "Unsupported OS."

spotify_path = find_spotify_path()
print(f"Spotify path: {spotify_path}")

import platform

def get_active_window_process_name_win():

    try:
        pid = win32process.GetWindowThreadProcessId(win32gui.GetForegroundWindow())
        active_window_process_name = psutil.Process(pid[-1]).name()
        return active_window_process_name
    except Exception as e:
        return "Error: " + str(e)

# def get_active_window_process_name_mac():
#
#     try:
#         ws = AppKit.NSWorkspace.sharedWorkspace()
#         active_app = ws.frontmostApplication()
#         active_app_name = active_app.localizedName()
#         return active_app_name
#     except Exception as e:
#         return "Error: " + str(e)

if platform.system() == 'Windows':
    print("Active window process name:", get_active_window_process_name_win())
# elif platform.system() == 'Darwin':  # macOS is identified as 'Darwin'
#     print("Active window process name:", get_active_window_process_name_mac())
else:
    print("Unsupported OS")