import time
import sys
import threading
import queue
import json
import os
from pynput import keyboard
import re

sys.path.append('../')
from qysea.sdk.manage import QY_Rov_Manage

class QyseaRovStatus:
    def __init__(self, fps=2, start_episode=0):
        self.fps = fps
        self.status_manage = self.connect_to_rov()
        self.status_queue = queue.Queue(maxsize=100)
        self.recording = False
        self.stopped = False
        self.save_path = None
        self.episode_counter = start_episode
        self.latest_status = None

    def load_latest_status(self):
        while True:
            self.latest_status = self.status_manage.get_rov_status()
            if self.latest_status['status_code'] != '200':
                print('Failed to get status')
                # break
            time.sleep(0.1)

    def connect_to_rov(self):
        print('Connecting to ROV...')
        connect_status = {'text': 'Failed to connect to ROV'}
        while "Success" not in connect_status['text']:
            status_manage = QY_Rov_Manage.QYRovRealTimeStatusManage()
            connect_status = status_manage.connect_to_rov()
            time.sleep(0.2)
            print("Connecting to ROV...")
        if "Success" in connect_status['text']:
            print("Connected to ROV")
        else:
            print("Failed to connect to ROV")
        time.sleep(1)
        return status_manage

    def read_status(self):
        interval = 1 / self.fps
        while not self.stopped:
            start_time = time.perf_counter()
            status = self.status_manage.get_rov_status()
            if status:
                if status['status_code'] != '200':
                    print('Failed to get status (read_status)')
                timestamp = time.perf_counter()
                if self.recording:
                    self.status_queue.put_nowait((status, timestamp))
            elapsed_time = time.perf_counter() - start_time
            sleep_time = max(0, interval - elapsed_time)
            time.sleep(sleep_time)

    def save_status(self):
        while not self.stopped or not self.status_queue.empty():
            try:
                status, timestamp = self.status_queue.get_nowait()
                if self.recording:
                    filename = os.path.join(self.save_path, f"{int(timestamp * 1000)}.json")
                    with open(filename, 'w') as f:
                        json.dump(status, f)
            except queue.Empty:
                time.sleep(0.01)

    def start(self):
        self.stopped = False
        threading.Thread(target=self.read_status, daemon=True).start()
        threading.Thread(target=self.save_status, daemon=True).start()

    def stop(self):
        self.stopped = True
        self.status_manage.disconnect_rov()
        print("Disconnected from ROV")

    def control_recording(self):
        def on_press(key):
            try:
                if key.char == 'c':
                    self.episode_counter += 1
                    self.save_path = os.path.join('data', f"episode_{self.episode_counter}", 'status')
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path, exist_ok=True)
                    time.sleep(0.1)
                    self.recording = True
                    print("Recording started (status) for episode", self.episode_counter)
                elif key.char == 's':
                    self.recording = False
                    print("Recording stopped (status) for episode", self.episode_counter)
                elif key.char == 'd': # no need to delete. Deletion is handled by all_cams.py
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
    rov_status = QyseaRovStatus(fps=2, start_episode=0)
    threading.Thread(target=rov_status.control_recording, daemon=True).start()
    rov_status.start()
    threading.Thread(target=rov_status.load_latest_status, daemon=True).start()
    try:
        while not rov_status.stopped:
            time.sleep(0.2)
    except KeyboardInterrupt:
        rov_status.stop()

if __name__ == "__main__":
    main()
