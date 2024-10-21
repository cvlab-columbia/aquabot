import cv2
import argparse
import os
import numpy as np

def save_frame(frame, save_dir, frame_index):
    filename = os.path.join(save_dir, f"frame-{frame_index}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Saved {filename}")

def main(video_left, video_right):
    cap1 = cv2.VideoCapture(video_left)
    cap2 = cv2.VideoCapture(video_right)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both videos.")
        return

    # Create directories for saving frames
    save_dir1 = os.path.join('cctv_calibration_videos', 'stereo', 'left')
    save_dir2 = os.path.join('cctv_calibration_videos', 'stereo', 'right')
    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)

    # Get the frame rate and total frame count
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    paused = False
    frame_index = 0
    saved_frame_index = 0

    # Set the desired frame rate
    desired_frame_rate = 10
    frame_interval = int(1000 / desired_frame_rate)  # in milliseconds

    def jump_frames(cap1, cap2, frames_to_jump):
        nonlocal frame_index
        frame_index += frames_to_jump
        frame_index = max(0, min(total_frames1 - 1, frame_index))
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    while True:
        if not paused:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                print("Reached the end of one or both videos.")
                break
            frame_index += 1

            # Get the dimensions of both frames
            height1, width1 = frame1.shape[:2]
            height2, width2 = frame2.shape[:2]

            # Create a blank canvas to hold both frames top and bottom
            combined_width = max(width1, width2)
            combined_height = height1 + height2
            combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

            # Place the frames on the canvas
            combined_frame[:height1, :width1] = frame1
            combined_frame[height1:height1+height2, :width2] = frame2

            # Resize to fit monitor height (assuming 1080p for this example)
            screen_height = 1080
            aspect_ratio = combined_width / combined_height
            new_height = screen_height
            new_width = int(aspect_ratio * new_height)
            combined_frame = cv2.resize(combined_frame, (new_width, new_height))

            # Show the combined frame
            cv2.imshow('Video', combined_frame)
            key = cv2.waitKey(frame_interval) & 0xFF
        else:
            # If paused, still check for key press but without delay
            key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            saved_frame_index += 1
            save_frame(frame1, save_dir1, saved_frame_index)
            save_frame(frame2, save_dir2, saved_frame_index)
        elif key == ord(' '):
            paused = not paused
            if paused:
                print("Video paused.")
            else:
                print("Video resumed.")
        elif key == ord('q'):
            break
        elif key == 83:  # right arrow key
            jump_frames(cap1, cap2, 10)  # forward 10 frames
        elif key == 81:  # left arrow key
            jump_frames(cap1, cap2, -10)  # backward 10 frames

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save frames from two videos when "c" is pressed and pause/resume with "space".')
    parser.add_argument('--left', type=str, required=True, help='Path to the first video file.')
    parser.add_argument('--right', type=str, required=True, help='Path to the second video file.')
    args = parser.parse_args()

    main(args.left, args.right)
