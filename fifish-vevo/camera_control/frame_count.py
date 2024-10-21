import cv2
import argparse

def count_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # Get the total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Release the video capture object
    video.release()
    
    print(f"Total number of frames: {total_frames}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count total number of frames in a video")
    parser.add_argument("--path", type=str, help="Path to the video file")
    args = parser.parse_args()
    
    count_frames(args.path)
