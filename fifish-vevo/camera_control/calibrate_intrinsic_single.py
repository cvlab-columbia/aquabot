import cv2
import numpy as np
import glob
import os

def calibrate_camera(image_folder, save_file_name, square_size):
    # Define the chessboard size
    chessboard_size = (5, 3)
    # Define the termination criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    # Get all images from the folder
    images = glob.glob(os.path.join(image_folder, '*.jpg'))

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(500)  # Display each image for 500ms

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        # Save the intrinsic matrix as a NumPy array
        np.save(save_file_name, mtx)
        print(f"Intrinsic matrix saved as {save_file_name}.npy")
    else:
        print("Calibration was unsuccessful.")

    cv2.destroyAllWindows()

# Example usage
# image_folder = 'cctv_calibration_videos/recording_cam0_20240730_152952'
# save_file_name = 'intrinsic_matrix_left'
image_folder = 'cctv_calibration_videos/recording_cam1_20240730_152952'
save_file_name = 'intrinsic_matrix_right'
square_size = 0.1  # Example: each square is 10cm or 0.1m

calibrate_camera(image_folder, save_file_name, square_size)
