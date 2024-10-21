import queue
import time
import cv2
import sys
import threading
import os
from pynput import keyboard

sys.path.append('../')
from qysea.sdk.manage import QY_Camera_Manage

class Camera_QYSEA:
    def __init__(self, name, fps):
        self.name = name
        self.fps = fps
        self.data_queue = queue.Queue(maxsize=100)
        self.display_queue = queue.Queue(maxsize=1)
        self.latest_frame = None
        self.recording = False
        self.stopped = False
        self.save_path = None

        # Initialize QYSEA camera parameters and media stream
        qy_cam_param_manage = QY_Camera_Manage.QYCameraParameterManage()
        print(qy_cam_param_manage.set_video_sub_stream_resolution(name, "480P"))
        print(qy_cam_param_manage.set_video_sub_stream_bitrate(name, "1M"))

        qy_cam_media_stream_manage = QY_Camera_Manage.QYCameraMediaStreamManage()
        print(qy_cam_media_stream_manage.capture_video(name, "SUB_STREAM"))

        self.qy_cam_media_stream_manage = qy_cam_media_stream_manage
        # self.video_handle = qy_cam_media_stream_manage.get_video_handle()

    def load_latest_frame(self):
        while True:
            if self.stopped:
                break
            try:
                ret, frame = self.qy_cam_media_stream_manage.get_frame()
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            # ret, frame = self.video_handle.read()
            if ret:
                self.latest_frame = frame
            time.sleep(0.01)

    def read_data(self):
        interval = 1 / self.fps
        while True:
            if self.stopped:
                break
            start_time = time.perf_counter()
            if self.latest_frame is not None and self.recording:
                timestamp = time.perf_counter()
                try:
                    self.data_queue.put_nowait((timestamp, self.latest_frame.copy()))
                    # print(f"Frame put in queue: timestamp={timestamp}")
                except queue.Full:
                    print("Queue is full, skipping frame")
            elapsed_time = time.perf_counter() - start_time
            sleep_time = max(0, interval - elapsed_time)
            # print(f"Read frame: elapsed_time={elapsed_time}, sleep_time={sleep_time}, queue_size={self.data_queue.qsize()}")
            time.sleep(sleep_time)

    def save_data(self):
        while True:
            save_path = self.save_path
            if self.stopped and self.data_queue.empty():
                break
            try:
                timestamp, frame = self.data_queue.get_nowait()
                # print(f"Frame retrieved from queue: timestamp={timestamp}")
            except queue.Empty:
                time.sleep(0.01)  # Short sleep to prevent busy waiting
                continue
            save_start_time = time.perf_counter()
            frame_resized = cv2.resize(frame, (640, 360))
            file_name = os.path.join(save_path, f"{int(timestamp * 1000)}.jpg")
            cv2.imwrite(file_name, frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            save_elapsed_time = time.perf_counter() - save_start_time
            # print(f"Saved frame: filename={file_name}, save_elapsed_time={save_elapsed_time}, queue_size={self.data_queue.qsize()}")

    def display_data(self):
        while True:
            if self.stopped:
                break
            if self.latest_frame is None:
                time.sleep(0.1)
                continue
            frame = self.latest_frame.copy()
            cv2.imshow(self.name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                self.stop()
                break
        cv2.destroyAllWindows()

    def on_press(self, key):
        try:
            if key.char == 'c':
                self.recording = True
                print("Started recording")
            elif key.char == 's':
                self.recording = False
                print("Stopped recording")
            elif key.char == 'q':
                self.stop()
                return False
        except AttributeError:
            pass

    def start(self):
        """
        Start the reading, saving, and displaying process in separate threads.
        """
        self.stopped = False
        threading.Thread(target=self.load_latest_frame, daemon=True).start()
        threading.Thread(target=self.read_data, daemon=True).start()
        threading.Thread(target=self.save_data, daemon=True).start()

    def stop(self):
        """
        Stop the reading, saving, and displaying process.
        """
        self.stopped = True
        cv2.destroyAllWindows()

    def control_recording(self):
        """
        Control recording with 'c' to start and 's' to stop.
        """
        def on_press(key):
            try:
                if key.char == 'c':
                    self.recording = True
                    print("Recording started.")
                elif key.char == 's':
                    self.recording = False
                    print("Recording stopped.")
                elif key.char == 'q':
                    self.stop()
                    time.sleep(0.1)
                    return False  # Stop listener
            except AttributeError:
                pass  # Handle special keys like shift, etc.

        # Collect events until released
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

def main():
    # Initialize the sensor
    camera_sensor = Camera_QYSEA("MAIN_CAMERA", 10)  # 10 FPS

    camera_sensor.save_path = 'rov_main'
    threading.Thread(target=camera_sensor.control_recording, daemon=True).start()

    # Start reading data
    camera_sensor.start()

    threading.Thread(target=camera_sensor.display_data, daemon=True).start()

    # Run until stopped
    while not camera_sensor.stopped:
        time.sleep(1)
    camera_sensor.qy_cam_media_stream_manage.close_video()

if __name__ == "__main__":
    main()
