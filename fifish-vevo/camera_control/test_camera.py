import skvideo.io
import cv2

# Set up video capture using skvideo
cap = skvideo.io.vreader("/dev/video1")  # Use appropriate device file for your camera

while True:
    try:
        frame = next(cap)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert color format to match OpenCV's BGR format
        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
            break
    except StopIteration:
        print("End of video stream")
        break

cv2.destroyAllWindows()
