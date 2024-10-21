import os
import cv2
import torch
import argparse
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import skvideo.io
import threading
from robot_detector import RobotDetector
import sys
import time
sys.path.append('../rov_control')
from cctv_camera import Camera_CCTV
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "/home/rliu/miniforge3/envs/zima-blue/lib/python3.10/site-packages/PyQt5/Qt/plugins"

class VideoStreamWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('All Cameras')
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        self.frame = None

    def set_frame(self, frame):
        self.frame = frame
        self.update_frame()

    def update_frame(self):
        if self.frame is not None:
            height, width, channel = self.frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(self.frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))


class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.robot_detector = RobotDetector(checkpoint_path='../object_detection/checkpoints/checkpoint_10.pth',
                                            calibration_path='../camera_control/calibration_res_07-30', device='cuda:1')
        self.cctv_left = Camera_CCTV(video_path="/dev/video1", fps=10)
        self.cctv_right = Camera_CCTV(video_path="/dev/video3", fps=10)

    def run(self):
        threading.Thread(target=self.cctv_left.load_latest_frame, daemon=True).start()
        threading.Thread(target=self.cctv_right.load_latest_frame, daemon=True).start()

        screen_width = 1920
        screen_height = 1080

        while True:
            time.sleep(0.02)
            try:
                if self.cctv_left.latest_frame is not None and self.cctv_right.latest_frame is not None:

                    frame_left = self.cctv_left.latest_frame.copy()
                    frame_right = self.cctv_right.latest_frame.copy()

                    try:
                        position_3d , p1, p2 = self.robot_detector.detect_robot_3d(frame_left, frame_right, night=False, return_2d=True)

                        print('3D position:', position_3d[0], position_3d[1], position_3d[2])

                        cv2.circle(frame_left, tuple(p1), radius=5, color=(0, 255, 0), thickness=-1)
                        cv2.circle(frame_right, tuple(p2), radius=5, color=(0, 255, 0), thickness=-1)

                        frames = [frame_left, frame_right]

                    except ValueError as e:
                        print(e)

                    # frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
                    # frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)

                    frames = [frame_left, frame_right]

                    combined_frame = cv2.hconcat(frames)
                    height, width, _ = combined_frame.shape
                    scale_factor = min(screen_width / width, screen_height / height)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)

                    resized_frame = cv2.resize(combined_frame, (new_width, new_height))

                    self.frame_ready.emit(resized_frame)
                else:
                    print("No frames yet")
            except Exception as e:
                print(e)


def main():
    app = QApplication(sys.argv)
    video_widget = VideoStreamWidget()

    video_thread = VideoThread()
    video_thread.frame_ready.connect(video_widget.set_frame)
    video_thread.start()

    video_widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
