import cv2
import os
import sys
sys.path.append('../')
import threading
import time
from cctv_camera import Camera_CCTV
from qysea_camera import Camera_QYSEA
from pynput import keyboard
import shutil

# Define video sources
video_sources = {
    "/dev/video1": "cctv_left",
    "/dev/video2": "rov_mount",
    "/dev/video3": "cctv_right"
}

# Initialize cameras
cctv_left = Camera_CCTV(video_path="/dev/video1", fps=10)
rov_mount = Camera_CCTV(video_path="/dev/video2", fps=10)
cctv_right = Camera_CCTV(video_path="/dev/video3", fps=10)
rov_main = Camera_QYSEA(name="MAIN_CAMERA", fps=10)

cameras = [cctv_left, rov_mount, cctv_right, rov_main]

# Global recording state and episode counter
recording = False
episode_counter = 0 # change this to starting episode number - 1
base_path = "data"

def on_press(key):
    global recording, episode_counter, base_path, cameras
    try:
        if key.char == 'c':
            episode_counter += 1
            for cam, data_name in zip(cameras, ['cctv_left', 'rov_mount', 'cctv_right', 'rov_main']):
                episode_path = os.path.join(base_path, f"episode_{episode_counter}", data_name)
                cam.save_path = episode_path
                print(cam.save_path)
                if not os.path.exists(episode_path):
                    os.makedirs(episode_path)
            time.sleep(0.1)
            for cam in cameras:
                cam.recording = True
            print("Recording started (cameras) for episode", episode_counter)
        elif key.char == 's':
            for cam in cameras:
                cam.recording = False
            print("Recording stopped (cameras) for episode", episode_counter)
        elif key.char == 'd':
            shutil.rmtree(os.path.join(base_path, f"episode_{episode_counter}"))
            print("Deleted episode", episode_counter)
            episode_counter -= 1
        elif key.char == 'q':
            for cam in cameras:
                cam.stop()
            return False  # Stop listener
    except AttributeError:
        pass

def display_latest_frames():
    while True:
        start_time = time.perf_counter()
        frames = []
        for cam, data_name in zip(cameras, ['cctv_left', 'rov_mount', 'cctv_right', 'rov_main']):
            if cam.latest_frame is not None:
                frame = cam.latest_frame.copy()
                # resize to 360p
                frame = cv2.resize(frame, (640, 360))
                if data_name != 'rov_main':
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame)

        if len(frames) == 4:
            # concatenate into 2x2 grid
            combined_frame = cv2.vconcat([cv2.hconcat(frames[:2]), cv2.hconcat(frames[2:])])
            # resize to fit on screen
            combined_frame = cv2.resize(combined_frame, (1280, 720))
            cv2.imshow("All Cameras", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            for cam in cameras:
                cam.stop()
            break
        elapsed_time = time.perf_counter() - start_time
        # print(f"Time taken to display frames: {elapsed_time:.3f} seconds")

    cv2.destroyAllWindows()

def start_cameras():
    global episode_counter, base_path
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    for cam, data_name in zip(cameras, ['cctv_left', 'rov_mount', 'cctv_right', 'rov_main']):
        episode_path = os.path.join(base_path, f"episode_{episode_counter}", data_name)
        cam.save_path = episode_path
        cam.start()


# Listener for keyboard events
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Start camera threads
start_cameras()

# Start display thread
threading.Thread(target=display_latest_frames, daemon=True).start()

# Keep the main thread running
listener.join()
