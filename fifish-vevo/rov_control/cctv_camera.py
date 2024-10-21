import skvideo.io
import cv2
import os
import time
import threading
from pynput import keyboard
from queue import Queue, Full, Empty

class Camera_CCTV:
    def __init__(self, video_path, fps=30):
        self.video_path = video_path
        self.fps = fps
        self.queue = Queue(maxsize=100)  # Adjust the max size as needed
        self.cap = skvideo.io.vreader(video_path)
        self.name = video_path
        self.stopped = False
        self.recording = False
        self.latest_frame = None
        self.save_path = None

    def load_latest_frame(self):
        """
        Load the latest frame from the video stream.
        """
        for frame in self.cap:
            if self.stopped:
                break
            self.latest_frame = frame
            time.sleep(0.01)  # Short sleep to prevent busy waiting

    def read_data(self):
        """
        Read video data at the specified fps and put frames into a queue.
        """
        interval = 1 / self.fps
        while True:
            start_time = time.perf_counter()
            if self.stopped:
                break
            if self.latest_frame is not None:
                frame_time = time.perf_counter()
                frame = self.latest_frame.copy()
                if self.recording:
                    self.queue.put_nowait((frame, frame_time))
            elapsed_time = time.perf_counter() - start_time
            sleep_time = max(0, interval - elapsed_time)
            time.sleep(sleep_time)
            # print(f"Read frame: elapsed_time={elapsed_time}, sleep_time={sleep_time}, queue_size={self.queue.qsize()}")

    def save_data(self):
        """
        Save data from the queue, resizing images to 360p and saving them as jpg with compression.
        """
        while True:
            save_path = self.save_path
            # os.makedirs(save_path, exist_ok=True)
            if self.stopped and self.queue.empty():
                break
            try:
                frame, timestamp = self.queue.get_nowait()
            except Empty:
                time.sleep(0.01)  # Short sleep to prevent busy waiting
                continue
            save_start_time = time.perf_counter()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            resized_frame = cv2.resize(frame, (640, 360))
            filename = f"{save_path}/{int(timestamp * 1000)}.jpg"
            cv2.imwrite(filename, resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            save_elapsed_time = time.perf_counter() - save_start_time
            # print(f"Saved {filename} in {save_elapsed_time:.6f} seconds.")

    def display_data(self):
        while True:
            if self.stopped:
                break
            if self.latest_frame is None:
                time.sleep(0.1)
                continue
            frame = self.latest_frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.video_path, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                cv2.destroyAllWindows()
                break
        cv2.destroyAllWindows()

    def start(self):
        """
        Start the reading, saving, and displaying processes in separate threads.
        """
        self.stopped = False
        threading.Thread(target=self.load_latest_frame, daemon=True).start()
        threading.Thread(target=self.read_data, daemon=True).start()
        threading.Thread(target=self.save_data, daemon=True).start()


    def stop(self):
        """
        Stop the reading, saving, and displaying processes.
        """
        self.stopped = True

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

if __name__ == "__main__":
    cctv = Camera_CCTV(video_path='/dev/video1', fps=10)
    threading.Thread(target=cctv.control_recording, daemon=True).start()
    cctv.save_path = 'cctv_left'
    cctv.start()
    threading.Thread(target=cctv.display_data, daemon=True).start()
    while not cctv.stopped:
        time.sleep(1)
