import skvideo.io
import numpy as np
import cv2
from datetime import datetime
import threading
import os
import time

# Get screen resolution
screen_width = 1920
screen_height = 1080

# Create directory for storing combined video
os.makedirs('combined', exist_ok=True)

# Open the video streams
cap1 = skvideo.io.vreader("/dev/video1")
cap2 = skvideo.io.vreader("/dev/video3")

# Global variables to store the latest frames
latest_frame1 = None
latest_frame2 = None

# Lock for synchronizing access to the latest frames
frame_lock = threading.Lock()

def capture_video_stream(video_reader, stream_id):
    global latest_frame1, latest_frame2
    for frame in video_reader:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with frame_lock:
            if stream_id == 1:
                latest_frame1 = frame_bgr
            elif stream_id == 2:
                latest_frame2 = frame_bgr

# Start threads to read video streams
thread1 = threading.Thread(target=capture_video_stream, args=(cap1, 1))
thread2 = threading.Thread(target=capture_video_stream, args=(cap2, 2))

thread1.daemon = True
thread2.daemon = True

thread1.start()
thread2.start()

# Function to initialize the video writer
def initialize_video_writer():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"combined/capture_{timestamp}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    writer = cv2.VideoWriter(filename, fourcc, 30.0, (960*2, 540))  # Combined width (2 x 960) and 540p height

    return writer

# Flag to check if recording is in progress
recording = False
video_writer = None

# Set target frame rate
target_frame_rate = 30.0
frame_duration = 1.0 / target_frame_rate

# Main loop to display images and save video
print("Press 'c' to start recording. Recording will stop automatically after 1 minute.")
start_time = None

while True:
    loop_start_time = time.time()

    with frame_lock:
        if latest_frame1 is not None and latest_frame2 is not None:
            # Resize frames to 540p resolution
            resized_frame1 = cv2.resize(latest_frame1, (960, 540))
            resized_frame2 = cv2.resize(latest_frame2, (960, 540))

            # Combine frames for display and recording
            combined_frame = cv2.hconcat([resized_frame1, resized_frame2])
            cv2.imshow('Video Stream', combined_frame)

            # Write frames to video file if recording
            if recording and video_writer is not None:
                video_writer.write(combined_frame)

            # Check if recording time has reached 1 minute
            if recording and (time.time() - start_time >= 60):
                recording = False
                video_writer.release()
                video_writer = None
                print("Recording stopped automatically after 1 minute.")
                break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and not recording:
        # Start recording
        recording = True
        video_writer = initialize_video_writer()
        start_time = time.time()
        print("Recording started.")
    elif key == ord('q'):
        break

    # Ensure loop adheres to target frame rate
    loop_end_time = time.time()
    elapsed_time = loop_end_time - loop_start_time
    if elapsed_time < frame_duration:
        time.sleep(frame_duration - elapsed_time)

# Release the video writer and close windows
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()

# Join threads to ensure cleanup
thread1.join()
thread2.join()
