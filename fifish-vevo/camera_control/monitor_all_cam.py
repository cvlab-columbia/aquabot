import skvideo.io
import numpy as np
import cv2
from datetime import datetime
import threading
import os

# Get screen resolution for display
display_width = 1920
display_height = 1080

# Resolution for recording
recording_width = 2560
recording_height = 1440

# Open the video streams
caps = [
    skvideo.io.vreader("/dev/video0"),
    skvideo.io.vreader("/dev/video1"),
    skvideo.io.vreader("/dev/video2"),
    skvideo.io.vreader("/dev/video3")
]

# Global variables to store the latest frames and recording state
latest_frames = [None, None, None, None]
recording = False
video_writer = None

def capture_video_stream(video_reader, stream_id):
    global latest_frames
    for frame in video_reader:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        latest_frames[stream_id] = frame_bgr

# Start threads to read video streams
threads = []
for i in range(4):
    thread = threading.Thread(target=capture_video_stream, args=(caps[i], i))
    thread.start()
    threads.append(thread)

# Function to start or stop recording
def toggle_recording():
    global recording, video_writer
    if recording:
        video_writer.release()
        video_writer = None
        recording = False
        print("Recording stopped.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.mp4"
        video_writer = skvideo.io.FFmpegWriter(filename, outputdict={'-r': '30', '-s': f'{recording_width}x{recording_height}'})
        recording = True
        print(f"Recording started: {filename}")

# Main loop to display images on key press 'c'
print("Press 'c' to start/stop recording and 'q' to quit.")
while True:
    if all(frame is not None for frame in latest_frames):
        # Resize frames for recording
        recording_frames = [cv2.resize(frame, (recording_width // 2, recording_height // 2)) for frame in latest_frames]
        recording_top_row = cv2.hconcat(recording_frames[:2])
        recording_bottom_row = cv2.hconcat(recording_frames[2:])
        recording_combined_frame = cv2.vconcat([recording_top_row, recording_bottom_row])

        # Resize frames for display
        display_frames = [cv2.resize(frame, (display_width // 2, display_height // 2)) for frame in latest_frames]
        display_top_row = cv2.hconcat(display_frames[:2])
        display_bottom_row = cv2.hconcat(display_frames[2:])
        display_combined_frame = cv2.vconcat([display_top_row, display_bottom_row])
        
        cv2.imshow('Video Stream', display_combined_frame)

        if recording:
            video_writer.writeFrame(cv2.cvtColor(recording_combined_frame, cv2.COLOR_BGR2RGB))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        toggle_recording()
    elif key == ord('p'):
        break

# Release the video streams and close windows
cv2.destroyAllWindows()

# Join threads to ensure cleanup
for thread in threads:
    thread.join()

# Release video writer if recording
if recording:
    video_writer.release()
