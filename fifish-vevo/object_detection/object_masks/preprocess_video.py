import cv2
import numpy as np
import os

input_video_path = 'background_removed.mp4'

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file {input_video_path}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a directory to store the frames with transparent background
frame_dir = 'frames'
os.makedirs(frame_dir, exist_ok=True)

# Define the erosion and dilation kernels
kernel = np.ones((3, 3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for green color and create mask
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply erosion to remove small clusters
    mask = cv2.erode(mask, kernel, iterations=1)

    # Apply dilation to restore object size
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Invert mask to get foreground
    mask_inv = cv2.bitwise_not(mask)

    # Extract the foreground
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Add alpha channel to the frame
    b, g, r = cv2.split(fg)
    alpha = mask_inv
    rgba = cv2.merge((b, g, r, alpha))

    # Find the bounding box of the non-green area
    coords = cv2.findNonZero(mask_inv)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the frame to the bounding box
    cropped_frame = rgba[y:y+h, x:x+w]

    # Save the cropped frame as PNG
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    frame_filename = os.path.join(frame_dir, f'frame_{frame_number:04d}.png')
    cv2.imwrite(frame_filename, cropped_frame)

# Release resources
cap.release()

print(f"Video processing complete. Frames saved in '{frame_dir}'")
