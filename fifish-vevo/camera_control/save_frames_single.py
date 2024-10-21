import cv2
import argparse
import os

def save_frame(frame, save_dir, frame_index):
    filename = os.path.join(save_dir, f"frame-{frame_index}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Saved {filename}")

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create a directory with the same name as the video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(os.path.dirname(video_path), video_name)
    os.makedirs(save_dir, exist_ok=True)

    # Get the frame rate and total frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    paused = False
    frame_index = 0
    saved_frame_index = 0

    # Set the desired frame rate
    desired_frame_rate = 10
    frame_interval = int(1000 / desired_frame_rate)  # in milliseconds

    def jump_frames(frames_to_jump):
        nonlocal frame_index
        frame_index += frames_to_jump
        frame_index = max(0, min(total_frames - 1, frame_index))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Reached the end of the video.")
                break
            frame_index += 1

            # Show frame only at the desired frame rate
            cv2.imshow('Video', frame)
            key = cv2.waitKey(frame_interval) & 0xFF
        else:
            # If paused, still check for key press but without delay
            key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            saved_frame_index += 1
            save_frame(frame, save_dir, saved_frame_index)
        elif key == ord(' '):
            paused = not paused
            if paused:
                print("Video paused.")
            else:
                print("Video resumed.")
        elif key == ord('q'):
            break
        elif key == 83:  # right arrow key
            jump_frames(int(2 * fps))  # forward 2 seconds
        elif key == 81:  # left arrow key
            jump_frames(-int(2 * fps))  # backward 2 seconds

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save frames from video when "c" is pressed and pause/resume with "space".')
    parser.add_argument('--path', type=str, required=True, help='Path to the video file.')
    args = parser.parse_args()

    main(args.path)
