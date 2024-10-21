from qysea_deploy_sl import QyseaDeployment
from learning import *
from pynput import keyboard
import os
import cv2
import torch
import threading
import time
import numpy as np
from flask import Flask, Response

base_dir = '../../policy/checkpoints/'
model_dir = 'n_obs_2_n_pred_2_interval_100_batch_size_32_lr_0.0001_loss_mse_seperate_encoder_True_status_conditioned_False__obs_interval_300_bottleneck-dim-None_vae_weight-0.0'
checkpoint = 'best_modelm.pth'
checkpoint_dir = os.path.join(base_dir, model_dir, checkpoint)

# Loading policy
robot_deployment = QyseaDeployment(checkpoint_dir, n_obs=2, n_pred=2, delay=0, interval=100, obs_interval=300, bottleneck_dim=None)

# Global variable to hold the current frame
current_frame = None
frame_lock = threading.Lock()  # Lock to ensure thread-safe access to current_frame

# Flask app for video streaming
app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(stream_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

def stream_video():
    global current_frame
    while True:
        with frame_lock:
            if current_frame is not None:
                # Encode the frame in JPEG format
                ret, jpeg = cv2.imencode('.jpg', current_frame)
                if not ret:
                    continue
                frame = jpeg.tobytes()
                # Return the frame to be streamed
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.02)  # Adjust the sleep time as needed

# Wait until the spacebar ('s') is pressed
def wait_for_spacebar():
    with keyboard.Events() as events:
        for event in events:
            if isinstance(event, keyboard.Events.Press) and event.key == keyboard.Key.space:
                break

def display_images(robot_deployment):
    global current_frame
    while True:
        try:
            latest_frames = {
                'cctv_left': robot_deployment.cctv_left.latest_frame.copy() if robot_deployment.cctv_left.latest_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                'cctv_right': robot_deployment.cctv_right.latest_frame.copy() if robot_deployment.cctv_right.latest_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                'rov_mount': robot_deployment.rov_mount.latest_frame.copy() if robot_deployment.rov_mount.latest_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                'rov_main': robot_deployment.rov_main.latest_frame.copy() if robot_deployment.rov_main.latest_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
            }
            
            # Prepare frames for display
            frames = []
            for key in ['cctv_left', 'rov_mount', 'cctv_right', 'rov_main']:
                frame = latest_frames[key]
                frame = cv2.resize(frame, (640, 360))
                if key != 'rov_main':
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame)
            
            if len(frames) == 4:
                combined_frame = cv2.vconcat([cv2.hconcat(frames[:2]), cv2.hconcat(frames[2:])])
                combined_frame = cv2.resize(combined_frame, (1280, 720))
                
                with frame_lock:  # Ensure thread-safe access
                    current_frame = combined_frame
            else:
                print("Frames are not being captured correctly.")
        except Exception as e:
            print('Error displaying images:', e)

        time.sleep(0.02)  # Small delay to prevent overwhelming the CPU

# Function to run the display in a separate thread
def start_display_thread(robot_deployment):
    display_thread = threading.Thread(target=display_images, args=(robot_deployment,))
    display_thread.daemon = True  # Ensures the thread will exit when the main program does
    display_thread.start()
    return display_thread

# Function to start the Flask server
def start_flask_app():
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False))
    flask_thread.daemon = True
    flask_thread.start()


# Start loading sensors
robot_deployment.start_loading(display_thread=False)

# Start the display thread
display_thread = start_display_thread(robot_deployment)

# Start the Flask app
start_flask_app()

for i in range(1000):
    robot_deployment.run_exp_garbage(frequency=10, duration=10000)  # Control frequency, duration