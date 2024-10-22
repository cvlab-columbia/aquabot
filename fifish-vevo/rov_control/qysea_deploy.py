import sys
import os
import time
import threading
import torch
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QEvent
from multiprocessing.connection import Client
import subprocess
import json
from utils import get_distances

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "/home/rliu/miniforge3/envs/zima-blue/lib/python3.10/site-packages/PyQt5/Qt/plugins"

from cctv_camera import Camera_CCTV
from qysea_camera import Camera_QYSEA
from qysea_status import QyseaRovStatus

import albumentations as A
from albumentations.pytorch import ToTensorV2

class test_preprocess:
    def __init__(self, img_size):
        self.img_size = img_size
        self.test_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def preprocess_image(self, frame):
        augmented = self.test_transform(image=frame)
        frame = augmented['image']
        return frame

sys.path.append('../../policy/')
from network import RobotPolicyModel

img_size = 224
slices = [0, 1, 2, 3, 4, 5, 6]

class VideoStreamWidget(QWidget):
    def __init__(self, model_dir, scaling_factor):
        super().__init__()
        self.setWindowTitle('All Cameras')
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        self.frame = None
        self.recording = False
        self.video_writer = None
        self.model_dir = model_dir
        self.episode_count = 0
        self.scaling_factor = [1.2, 1.2, 1.2, 1.2]
        self.date_time_str = time.strftime("%Y%m%d-%H%M%S")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.start_recording()
        elif event.key() == Qt.Key_M:
            self.stop_recording()

    def start_recording(self):
        scaling_str = '-'.join([str(int(s)) for s in self.scaling_factor])
        if not self.recording:
            os.makedirs(f'experiments/date-{self.date_time_str}-{self.model_dir}-scaling-{scaling_str}', exist_ok=True)
            self.episode_count += 1
            video_path = f'experiments/date-{self.date_time_str}-{self.model_dir}-scaling-{scaling_str}/episode_{self.episode_count}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 10, (1280, 720))
            self.recording = True
            print(f"Started recording: {video_path}")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.video_writer.release()
            self.video_writer = None
            print(f"Stopped recording: episode_{self.episode_count}")

    def set_frame(self, frame):
        self.frame = frame
        self.update_frame()

    def update_frame(self):
        if self.frame is not None:
            frame_display = cv2.resize(self.frame.copy(), (1280, 720))
            height, width, channel = frame_display.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame_display.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
            if self.recording and self.video_writer is not None:
                self.video_writer.write(cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))

class QyseaDeployment:
    def __init__(self, checkpoint, n_obs=3, n_pred=8, delay=1, interval=100, obs_interval=400, \
                 qysea_xbox_address=('localhost', 8080), scaling_factor=[1., 1., 1., 1.], bottleneck_dim=None):
        self.n_obs = n_obs
        self.delay = delay
        self.interval = interval
        self.obs_interval = obs_interval
        self.total_frames = n_obs * (obs_interval // interval)
        self.qysea_xbox_address = qysea_xbox_address
        self.scaling_factor = scaling_factor
        self.slices = [(self.obs_interval // self.interval) * i - 1 for i in range(1, self.n_obs + 1)] # slices for observation steps
        self.states_buffer = {
            'cctv_left': torch.zeros((1, self.total_frames, 3, img_size, img_size), device='cuda'),
            'cctv_right': torch.zeros((1, self.total_frames, 3, img_size, img_size), device='cuda'),
            'rov_main': torch.zeros((1, self.total_frames, 3, img_size, img_size), device='cuda'),
            'rov_mount': torch.zeros((1, self.total_frames, 3, img_size, img_size), device='cuda'),
            'action': torch.zeros((1, self.total_frames, len(slices)), device='cuda'),
            'status': torch.zeros((1, self.total_frames, 3), device='cuda')
        }

        # Initialize cameras and status
        self.cctv_left = Camera_CCTV(video_path="/dev/video1", fps=10)
        self.cctv_right = Camera_CCTV(video_path="/dev/video3", fps=10)
        self.rov_mount = Camera_CCTV(video_path="/dev/video2", fps=10)
        self.rov_main = Camera_QYSEA(name="MAIN_CAMERA", fps=10)
        self.qysea_status = QyseaRovStatus(fps=2)
        time.sleep(1)

        self.policy_network = RobotPolicyModel(n_obs, n_pred, action_contidioned=False, seperate_encoder=True, \
                                            status_conditioned=False, bottleneck_dim=bottleneck_dim)
        self.policy_network.load_state_dict(torch.load(checkpoint))
        self.policy_network = self.policy_network.to('cuda').eval()

        self.cameras = [self.cctv_left, self.cctv_right, self.rov_mount, self.rov_main]
        self.app = QApplication(sys.argv)
        self.video_widget = VideoStreamWidget(os.path.basename(os.path.dirname(checkpoint)), scaling_factor)
        self.video_widget.show()
        self.video_buffer = []
        self.action_buffer = []

        self.test_preprocess = test_preprocess(img_size)

        # self.qysea_xbox_proc = subprocess.Popen([sys.executable, 'qysea_xbox_subprocess.py', str(self.qysea_xbox_address)])

    def start_loading(self):
        for cam in self.cameras:
            threading.Thread(target=cam.load_latest_frame, daemon=True).start()
        threading.Thread(target=self.qysea_status.load_latest_status, daemon=True).start()

    def reward(self):
        '''
        calculate reward from self.video_buffer and self.action_buffer
        '''

        gripper_state_past = min(get_distances(self.video_buffer[-6]['rov_main']))
        gripper_state_now = min(get_distances(self.video_buffer[-1]['rov_main']))
        gripper_state_close = gripper_state_now < 0.1
        gripper_stuck = np.isclose(gripper_state_past, gripper_state_now, atol=0.02)
        
        # closing if 4 out of past 6 actions are closing gripper
        close_gripper = sum([action[5] > 0.38 for action in self.action_buffer[-6:]]) >= 4

        # print without newline
        print('gripper state: {0:.3f}'.format(gripper_state_now), 'closed:', gripper_state_close, 'stuck:', gripper_stuck, 'closing:', close_gripper, end=' ')

        # if closing gripper and gripper is not zero and gripper is not fully closed
        if close_gripper and not gripper_state_close and gripper_stuck:
            return 1
        else:
            return 0

    def policy_rollout(self, frequency=10):
        def rollout():
            max_retries = 20
            retry_delay = 1  # seconds
            retries = 0
            while retries < max_retries:
                try:
                    conn = Client(self.qysea_xbox_address)
                    conn.send('ping')
                    conn.close()
                    break
                except ConnectionRefusedError:
                    retries += 1
                    time.sleep(retry_delay)
            else:
                raise RuntimeError("Failed to connect to QyseaXbox subprocess after multiple attempts.")

            buffer_count = 0
            while True:
                if self.cctv_left.latest_frame is None or self.cctv_right.latest_frame is None or \
                        self.rov_mount.latest_frame is None or self.rov_main.latest_frame is None or \
                        self.qysea_status.latest_status is None or self.qysea_status.latest_status['status_code'] != '200':
                    time.sleep(0.1)
                    if self.cctv_left.latest_frame is None:
                        print('cctv_left frame is None')
                    if self.cctv_right.latest_frame is None:
                        print('cctv_right frame is None')
                    if self.rov_mount.latest_frame is None:
                        print('rov_mount frame is None')
                    if self.rov_main.latest_frame is None:
                        print('rov_main frame is None')
                    if self.qysea_status.latest_status is None:
                        print('status is None')
                    if self.qysea_status.latest_status['status_code'] != 200:
                        print('status code is not 200')
                    continue
                start_time = time.perf_counter()
                
                # Capture the latest frames
                latest_frames = {
                    'cctv_left': self.cctv_left.latest_frame.copy(),
                    'cctv_right': self.cctv_right.latest_frame.copy(),
                    'rov_main': self.rov_main.latest_frame.copy(),
                    'rov_mount': self.rov_mount.latest_frame.copy(),
                    'status': self.qysea_status.latest_status.copy()
                }

                # resize rov_mount due to new camera
                frame = latest_frames['rov_mount']
                h, w, _ = frame.shape
                frame = frame[0:int(h * 0.80), int(w * 0.1):int(w * 0.9)]
                frame = cv2.resize(frame, (640, 360))
                latest_frames['rov_mount'] = frame
                
                self.video_buffer.append(latest_frames.copy())

                try:
                    distances_left, distances_right = get_distances(latest_frames['rov_main'].copy())
                except:
                    distances_left, distances_right = 1., 1.
                print('Gripper states:', distances_left, distances_right)

                # Shift buffer to the left
                for key in self.states_buffer:
                    self.states_buffer[key] = torch.roll(self.states_buffer[key], shifts=-1, dims=1)
                
                self.states_buffer['cctv_left'][0, -1] = self.test_preprocess.preprocess_image(latest_frames['cctv_left'])
                self.states_buffer['cctv_right'][0, -1] = self.test_preprocess.preprocess_image(latest_frames['cctv_right'])
                self.states_buffer['rov_main'][0, -1] = self.test_preprocess.preprocess_image(latest_frames['rov_main'])
                self.states_buffer['rov_mount'][0, -1] = self.test_preprocess.preprocess_image(latest_frames['rov_mount'])
                self.states_buffer['status'][0, -1] = self.preprocess_status(latest_frames['status'])
                
                buffer_count += 1

                # slice the states buffer with self.slices
                input = {k: v[:, self.slices] for k, v in self.states_buffer.items()}

                # Predict action
                with torch.no_grad():
                    action = self.policy_network(input)
                
                pred_action_8 = action[0][self.delay].cpu().numpy()

                # scale the action
                for i in range(4):
                    pred_action_8[i] *= self.scaling_factor[i]

                pred_action_15 = np.zeros(15)
                for ii, i_action in enumerate(slices):
                    pred_action_15[i_action] = pred_action_8[ii]    

                # Send action to robot
                action_str = json.dumps(pred_action_15.tolist())
                with Client(self.qysea_xbox_address) as conn:
                    conn.send(action_str)
                
                print([round(p, 3) for p in pred_action_8])
                
                # inference time
                inference_time = time.perf_counter() - start_time
            
                self.states_buffer['action'][0, -1] = action[0][self.delay]
                self.action_buffer.append(pred_action_15)

                if len(self.action_buffer) > 20:
                    reward = self.reward()
                    print('Reward:', reward, end=' ')

                # Display frames
                frames = []
                for key in ['cctv_left', 'rov_mount', 'cctv_right', 'rov_main']:
                    frame = latest_frames[key]
                    frame = cv2.resize(frame, (640, 360))
                    if key != 'rov_main':
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frames.append(frame)
                
                if len(frames) == 4:
                    combined_frame = cv2.vconcat([cv2.hconcat(frames[:2]), cv2.hconcat(frames[2:])])
                    combined_frame = cv2.resize(combined_frame, (2560, 1440))
                    combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                    self.video_widget.set_frame(combined_frame)
                
                time_elapsed = time.perf_counter() - start_time
                if time_elapsed < 1 / frequency:
                    time.sleep(1 / frequency - time_elapsed)
                print('Rollout time:', time.perf_counter() - start_time, 'Inference time:', inference_time)
        threading.Thread(target=rollout, daemon=True).start()

    def preprocess_image(self, frame):
        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0 * 2. - 1.  # Normalize to [-1, 1]
        frame = torch.tensor(frame, dtype=torch.float32, device='cuda').permute(2, 0, 1)
        print(frame.min(), frame.max())
        return frame

    def preprocess_status(self, status):
        status_tensor = torch.tensor([status['Compass'], \
                                      status['Rov_Attitude_Angle']['Pitch'], \
                                      status['Rov_Attitude_Angle']['Roll']], dtype=torch.float32, device='cuda') / 360.
        return status_tensor

from pynput import keyboard

# Wait until the spacebar ('s') is pressed
def wait_for_spacebar():
    with keyboard.Events() as events:
        for event in events:
            if isinstance(event, keyboard.Events.Press) and event.key == keyboard.Key.space:
                break

if __name__ == '__main__':
    scaling_factor = [1.2, 1.2, 1.2, 1.2]
    base_dir = '../../policy/checkpoints/'
    model_dir = 'n_obs_2_n_pred_2_interval_100_batch_size_32_lr_0.0001_loss_mse_seperate_encoder_True_status_conditioned_False__obs_interval_300_bottleneck-dim-None_vae_weight-0.0_weight-decay-0.1'
    checkpoint = 'best_model.pth'

    checkpoint_dir = os.path.join(base_dir, model_dir, checkpoint)
    robot_deployment = QyseaDeployment(checkpoint_dir, n_obs=2, n_pred=2, delay=0, interval=100, obs_interval=300, scaling_factor=scaling_factor, bottleneck_dim=None)
    robot_deployment.start_loading()
    robot_deployment.policy_rollout(frequency=10)
    sys.exit(robot_deployment.app.exec_())
    while True:
        wait_for_spacebar()
        robot_deployment.reset()
    
