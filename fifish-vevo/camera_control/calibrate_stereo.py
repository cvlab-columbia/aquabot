import cv2
import numpy as np
import glob
import os

def stereo_calibrate(left_folder, right_folder, intrinsic_left_file, intrinsic_right_file, square_size):
    # Load intrinsic matrices
    intrinsic_matrix_left = np.load(intrinsic_left_file)
    intrinsic_matrix_right = np.load(intrinsic_right_file)

    # Define the chessboard size
    chessboard_size = (5, 3)
    # Define the termination criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d points in real world space
    imgpoints_left = []  # 2d points in image plane for left camera
    imgpoints_right = []  # 2d points in image plane for right camera

    # Get all images from the folders
    images_left = glob.glob(os.path.join(left_folder, '*.jpg'))
    images_right = glob.glob(os.path.join(right_folder, '*.jpg'))

    # Sort the images to ensure matching pairs
    images_left.sort()
    images_right.sort()

    for fname_left, fname_right in zip(images_left, images_right):
        img_left = cv2.imread(fname_left)
        img_right = cv2.imread(fname_right)
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners for left and right images
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

        if ret_left and ret_right:
            # Draw the corners on the images
            cv2.drawChessboardCorners(img_left, chessboard_size, corners_left, ret_left)
            cv2.drawChessboardCorners(img_right, chessboard_size, corners_right, ret_right)

            # Concatenate images vertically
            combined_img = cv2.vconcat([img_left, img_right])

            # Resize the image to fit within the monitor height
            screen_height = 1080  # Change this value based on your screen resolution
            scale_factor = screen_height / combined_img.shape[0]
            resized_img = cv2.resize(combined_img, (int(combined_img.shape[1] * scale_factor), screen_height))

            cv2.imshow('Chessboard Corners', resized_img)

            # Wait for user input to decide whether to use the frame
            while True:
                key = cv2.waitKey(0)
                if key == 13:  # Enter key
                    objpoints.append(objp)
                    corners_left2 = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                    imgpoints_left.append(corners_left2)
                    corners_right2 = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
                    imgpoints_right.append(corners_right2)
                    break
                elif key == 32:  # Space key
                    break

    # Destroy all the windows displaying images
    cv2.destroyAllWindows()

    # Stereo calibration
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        intrinsic_matrix_left, None,
        intrinsic_matrix_right, None,
        gray_left.shape[::-1], criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    if ret:
        print("Stereo calibration was successful.")
        print(f"Rotation matrix:\n{R}")
        print(f"Translation vector:\n{T}")
        np.save('rotation_matrix.npy', R)
        np.save('translation_vector.npy', T)
    else:
        print("Stereo calibration was unsuccessful.")

# Example usage
left_folder = 'cctv_calibration_videos/stereo/left'
right_folder = 'cctv_calibration_videos/stereo/right'
intrinsic_left_file = 'intrinsic_matrix_left.npy'
intrinsic_right_file = 'intrinsic_matrix_right.npy'
square_size = 0.1  # Each square is 10 cm or 0.1 meters

stereo_calibrate(left_folder, right_folder, intrinsic_left_file, intrinsic_right_file, square_size)
