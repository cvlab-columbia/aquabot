import skvideo.io
import time

def calculate_fps(video_source='/dev/video0', num_frames=100):
    # Open the video source
    video_capture = skvideo.io.vreader(video_source)
    
    frame_count = 0
    start_time = time.time()
    
    # Read and count the frames
    for frame in video_capture:
        frame_count += 1
        if frame_count >= num_frames:
            break
    
    end_time = time.time()
    
    # Calculate FPS
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time
    
    print(f"Captured {frame_count} frames in {elapsed_time:.2f} seconds.")
    print(f"Estimated FPS: {fps:.2f}")

if __name__ == "__main__":
    calculate_fps()
