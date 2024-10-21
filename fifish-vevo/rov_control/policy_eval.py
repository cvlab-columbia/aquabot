from qysea_deploy_eval import QyseaDeployment
from learning import *
from pynput import keyboard
import os
import cv2
import torch
import threading
import time
import numpy as np
from flask import Flask, Response


n_obs = 2
n_pred = 4
action_horizon = 1
epoch = 49

base_dir = '../../diffusion_policy/checkpoints_garbage/'
model_dir = f'obs_{n_obs}_pred_{n_pred}_bs_32'
checkpoint = f'checkpoint_epoch_40.pth'
policy_arch = 'dp'

# n_obs = 1
# model_dir = f'act/chunk_{n_pred}'
# checkpoint = f'policy_epoch_{epoch}_seed_42.ckpt'
# policy_arch = 'act'
# assert n_obs == 1

# base_dir = '../../policy/garbage_classification/08-30/checkpoints'
# model_dir = f'n_obs_{n_obs}_n_pred_{n_pred}_interval_100_batch_size_32_lr_0.0001_loss_mse_seperate_encoder_True_status_conditioned_False__obs_interval_300_bottleneck-dim-None_vae_weight-0.0'
# checkpoint = f'checkpoint_epoch_40.pth'
# policy_arch = 'mlp'

checkpoint_dir = os.path.join(base_dir, model_dir, checkpoint)

# Loading policy
robot_deployment = QyseaDeployment(checkpoint_dir, policy_arch, n_obs=n_obs, n_pred=n_pred, interval=100, obs_interval=300, bottleneck_dim=None, action_horizon=action_horizon)

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
def wait_for_r():
    with keyboard.Events() as events:
        for event in events:
            try:
                if isinstance(event, keyboard.Events.Press) and event.key.char == 'r':
                    break
            except AttributeError:
                pass

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

# Initialize learning
net = Net()

success_count = 0
time_taken = []

# Self-learning loop
for i_loop in range(20):
    # robot_deployment.scaling_factor = [1.4545, 1.795 , 1.674 , 1.967]
    robot_deployment.scaling_factor = [1.0, 1.0, 1.0, 1.0]

    # Reset rock
    reset_thread = threading.Thread(target=robot_deployment.reset, args=(10, [2.0, 1.8, 0.7], False))
    reset_thread.start()
    robot_deployment.wait_for_r()
    reset_thread.join()  # Wait for reset to complete
    robot_deployment.move_back(light=False)

    # Policy execution
    all_rewards, video = robot_deployment.run_exp(frequency=10, duration=60)  # Control frequency, duration

    # reward for shorter episodes (reward range: 0 to 30)
    if all_rewards is None: # return None if human has to intervene
        time_taken.append(-1)
    else:
        time_taken.append(len(all_rewards) / 10.)
        success_count += 1
    
    print('Time taken:', time_taken[-1])

print('Success rate:', success_count / 20)
valid_time_taken = [t for t in time_taken if t != -1]
print('Average time taken:', sum(valid_time_taken) / len(valid_time_taken))

model_name = os.path.join(model_dir, checkpoint).replace('/', '_').replace('.pth', '')
save_dir = os.path.join('icra_results')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# save txt
with open(os.path.join(save_dir, f'{model_name}.txt'), 'w') as f:
    f.write(f'Success rate: {success_count / 20}\n')
    f.write(f'Average time taken: {sum(valid_time_taken) / len(valid_time_taken)}\n')
    for time in time_taken:
        f.write(f'{time}\n')