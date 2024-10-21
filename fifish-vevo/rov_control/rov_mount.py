import cv2
import os
import sys
import threading
import time
from cctv_camera import Camera_CCTV
from pynput import keyboard

# Initialize the camera
rov_mount = Camera_CCTV(video_path="/dev/video2", fps=10)

# Global recording state and episode counter
recording = False
episode_counter = 0  # change this to starting episode number - 1
base_path = "data"

def on_press(key):
    global recording, episode_counter, base_path, rov_mount
    try:
        if key.char == 'c':
            episode_counter += 1
            episode_path = os.path.join(base_path, f"episode_{episode_counter}", "rov_mount")
            rov_mount.save_path = episode_path
            print(rov_mount.save_path)
            if not os.path.exists(episode_path):
                os.makedirs(episode_path)
            time.sleep(0.1)
            rov_mount.recording = True
            print("Recording started for episode", episode_counter)
        elif key.char == 's':
            rov_mount.recording = False
            print("Recording stopped for episode", episode_counter)
        elif key.char == 'd':
            shutil.rmtree(os.path.join(base_path, f"episode_{episode_counter}"))
            print("Deleted episode", episode_counter)
            episode_counter -= 1
        elif key.char == 'q':
            rov_mount.stop()
            return False  # Stop listener
    except AttributeError:
        pass

def display_latest_frame():
    while True:
        start_time = time.perf_counter()
        if rov_mount.latest_frame is not None:
            frame = rov_mount.latest_frame.copy()
            # Resize to 360p
            frame = cv2.resize(frame, (640, 360))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # crop 90% of the imageq
            h, w, _ = frame.shape
            frame = frame[0:int(h * 0.90), int(w * 0.05):int(w * 0.95)]

            cv2.imshow("ROV Mount Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rov_mount.stop()
            break
        elapsed_time = time.perf_counter() - start_time
        # print(f"Time taken to display frame: {elapsed_time:.3f} seconds")

    cv2.destroyAllWindows()

def start_camera():
    global episode_counter, base_path
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    episode_path = os.path.join(base_path, f"episode_{episode_counter}", "rov_mount")
    rov_mount.save_path = episode_path
    rov_mount.start()

# Listener for keyboard events
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Start the camera thread
start_camera()

# Start the display thread
threading.Thread(target=display_latest_frame, daemon=True).start()

# Keep the main thread running
listener.join()
