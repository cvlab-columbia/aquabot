import time
import sys
import threading
import queue
from pynput import keyboard
import numpy as np
import os
sys.path.append('../')
from qysea.sdk.manage import QY_Rov_Manage
from qysea.sdk.manage.QY_RovController_Manage import QYRovControllerManage
from xbox import XboxController
from concurrent.futures import ThreadPoolExecutor

class QyseaXbox:
    def __init__(self, fps=10, start_episode=0):
        self.fps = fps
        self.qy_rov_pa_manage = self.control_setting()
        self.qy_rov_controller_manage = self.connect_controller()
        self.xbox = self.initialize_joystick()
        self.control_queue = queue.Queue(maxsize=100)
        self.recording = False
        self.stopped = False
        self.latest_action = None
        self.save_path = None
        self.episode_counter = start_episode
        self.executor = ThreadPoolExecutor(max_workers=24)  # Adjust based on your system's capabilities


    def control_setting(self):
        print('====================')
        print('control setting:')
        qy_rov_pa_manage = QY_Rov_Manage.QYRovParameterManage()
        print('Set the controller operation mode to UAV_USA:')
        print(qy_rov_pa_manage.set_rov_controller_operation('UAV_USA'))
        print('Set the throttle curvature and limit: 0, 50')
        print(qy_rov_pa_manage.set_throttle_curvature_and_limit(0, 50))
        print(qy_rov_pa_manage.get_throttle_curvature_and_limit())
        print('Set the rotate curvature and limit: 0, 50')
        print(qy_rov_pa_manage.set_rotate_curvature_and_limit(0, 50))
        print(qy_rov_pa_manage.get_rotate_curvature_and_limit())
        print('Return to the original posture:')
        print(qy_rov_pa_manage.set_rov_posture_original())
        print('====================')
        return qy_rov_pa_manage

    def connect_controller(self):
        print('====================')
        print('connect to controller:')
        qy_rov_controller_manage = QYRovControllerManage()
        print(qy_rov_controller_manage.set_remote_control_status("ON"))
        qy_rov_controller_manage.set_right_wave_left_right(2000)
        time.sleep(2)
        qy_rov_controller_manage.set_right_wave_left_right(1500)
        print('====================')
        return qy_rov_controller_manage

    def initialize_joystick(self):
        print('Initialize and calibrating joystick:')
        return XboxController()

    def load_latest_action(self):
        while not self.stopped:
            self.latest_action = self.xbox.get_controller_input()
            time.sleep(0.005)

    def read_data(self):
        interval = 1 / self.fps
        while not self.stopped:
            start_time = time.perf_counter()
            if self.latest_action is not None:
                timestamp = time.perf_counter()
                action_copy = self.latest_action.copy()
                if not self.recording:
                    for i in range(5):
                        action_copy[i]# *= 1.5
                self.send_action(action_copy)
                if self.recording:
                    self.control_queue.put_nowait((action_copy, timestamp))
            elapsed_time = time.perf_counter() - start_time
            sleep_time = max(0, interval - elapsed_time)
            time.sleep(sleep_time)
            # print(f"elapsed_time: {elapsed_time}, sleep_time: {sleep_time}")

    def send_action(self, action):

        if np.isclose(action[5], 1., atol=0.05):
            # print('close gripper')
            self.qy_rov_controller_manage.set_right_wave_left_right(1000)
        elif np.isclose(action[6], 1., atol=0.05):
            # print('open gripper')
            self.qy_rov_controller_manage.set_right_wave_left_right(2000)
        else:
            # print('stable')
            self.qy_rov_controller_manage.set_right_wave_left_right(1500)

        self.qy_rov_controller_manage.set_left_joystick_left_right(int(action[0] * 200 + 1500))
        self.qy_rov_controller_manage.set_left_joystick_up_down(int(action[1] * 200 + 1500))
        self.qy_rov_controller_manage.set_right_joystick_left_right(int(action[2] * 200 + 1500))
        self.qy_rov_controller_manage.set_right_joystick_up_down(int(action[3] * 200 + 1500))
        self.qy_rov_controller_manage.set_left_wave_left_right(int(action[4] * 200 + 1500))

        if action[7] == 1:
            self.qy_rov_controller_manage.set_keep_depth_button(1)
        elif action[8] == 1:
            self.qy_rov_controller_manage.set_keep_depth_button(0)

        if action[9] == 1:
            self.qy_rov_controller_manage.set_rc_lock_button(0)
        elif action[10] == 1:
            self.qy_rov_controller_manage.set_rc_lock_button(1)

        if action[11] == 1:
            self.qy_rov_controller_manage.set_right_switch(0)
        elif action[12] == 1:
            self.qy_rov_controller_manage.set_right_switch(1)

        if action[13] == 1:
            self.qy_rov_pa_manage.set_rov_posture_original()


    def save_data(self):
        while not self.stopped or not self.control_queue.empty():
            try:
                action, timestamp = self.control_queue.get_nowait()
                np.save(os.path.join(self.save_path, f"{int(timestamp * 1000)}.npy"), action)
            except queue.Empty:
                time.sleep(0.01)

    def start(self):
        self.stopped = False
        threading.Thread(target=self.load_latest_action, daemon=True).start()
        threading.Thread(target=self.read_data, daemon=True).start()
        threading.Thread(target=self.save_data, daemon=True).start()

    def stop(self):
        self.stopped = True
        print(self.qy_rov_controller_manage.set_remote_control_status("OFF"))

    def control_recording(self):
        def on_press(key):
            try:
                if key.char == 'c':
                    self.episode_counter += 1
                    self.save_path = os.path.join('data', f"episode_{self.episode_counter}", 'action')
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path, exist_ok=True)
                    time.sleep(0.1)
                    self.recording = True
                    print("Recording started (action) for episode", self.episode_counter)
                elif key.char == 's':
                    self.recording = False
                    print("Recording stopped (action) for episode", self.episode_counter)
                elif key.char == 'd':
                    self.episode_counter -= 1
                elif key.char == 'q':
                    self.stop()
                    time.sleep(0.1)
                    return False
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

def main():
    qysea_xbox = QyseaXbox(fps=30, start_episode=0)
    threading.Thread(target=qysea_xbox.control_recording, daemon=True).start()
    qysea_xbox.start()
    while not qysea_xbox.stopped:
        time.sleep(0.2)

if __name__ == "__main__":
    main()
