import skvideo.io
import numpy as np
import cv2
from datetime import datetime
import threading
import os

# Get screen resolution
screen_width = 1920
screen_height = 1080

# Create directories for storing images
os.makedirs('left', exist_ok=True)
os.makedirs('right', exist_ok=True)

# Open the video streams
cap1 = skvideo.io.vreader("/dev/video1")
cap2 = skvideo.io.vreader("/dev/video3")

# Global variable to store the latest frames
latest_frame1 = None
latest_frame2 = None

def capture_video_stream(video_reader, stream_id):
    global latest_frame1, latest_frame2
    for frame in video_reader:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if stream_id == 1:
            latest_frame1 = frame_bgr
        elif stream_id == 2:
            latest_frame2 = frame_bgr

# Start threads to read video streams
thread1 = threading.Thread(target=capture_video_stream, args=(cap1, 1))
thread2 = threading.Thread(target=capture_video_stream, args=(cap2, 2))

thread1.start()
thread2.start()

# Function to capture and save images
def capture_images():
    global latest_frame1, latest_frame2
    if latest_frame1 is not None and latest_frame2 is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename1 = f"left/capture_{timestamp}_1.jpg"
        filename2 = f"right/capture_{timestamp}_2.jpg"
        
        cv2.imwrite(filename1, latest_frame1)
        cv2.imwrite(filename2, latest_frame2)
        
        print(f"Captured images saved as {filename1} and {filename2}")
    else:
        print("Error: Could not read frames from video streams")

# Main loop to display images on key press 'c'
print("Press 'c' to capture images and 'q' to quit.")
while True:
    if latest_frame1 is not None and latest_frame2 is not None:
        # Resize frames to fit within the screen resolution
        height, width, _ = latest_frame1.shape
        scale_factor = min(screen_width / (2 * width), screen_height / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        resized_frame1 = cv2.resize(latest_frame1, (new_width, new_height))
        resized_frame2 = cv2.resize(latest_frame2, (new_width, new_height))

        combined_frame = cv2.hconcat([resized_frame1, resized_frame2])
        cv2.imshow('Video Stream', combined_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        capture_images()
    elif key == ord('q'):
        break

# Release the video streams and close windows
cv2.destroyAllWindows()

# Join threads to ensure cleanup
thread1.join()
thread2.join()
