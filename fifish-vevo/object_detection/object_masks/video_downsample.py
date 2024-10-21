import cv2

# Open the input video file
input_video_path = 'robot.mp4'
output_video_path = 'robot_subsampled.mp4'

cap = cv2.VideoCapture(input_video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Cannot open video file {input_video_path}")
    exit()

# Get input video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate the interval for frame subsampling
interval = fps // 5  # 30fps to 5fps

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 5, (width, height))

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Write the frame if it's in the subsample interval
    if frame_idx % interval == 0:
        out.write(frame)
    
    frame_idx += 1

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video subsampled from 30fps to 5fps and saved as {output_video_path}")
