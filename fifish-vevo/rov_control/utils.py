import cv2
import numpy as np
import sys

def get_distances(image):

    # # robot 2
    # gripper_left = [0.321, 0.188] # close, open
    # gripper_right = [0.314, 0.184] # close, open

    # robot 1
    # gripper_left = [0.335, 0.203] # close, open
    # gripper_right = [0.295, 0.163] # close, open

    # robot 3
    gripper_left = [0.347, 0.215] # close, open
    gripper_right = [0.304, 0.167] # close, open

    height, width = image.shape[:2]
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the threshold to segment the gripper
    _, image_black = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Erosion and dilation to clean up the mask
    kernel = np.ones((8, 8), np.uint8)
    image_black = cv2.erode(image_black, kernel, iterations=1)
    image_black = cv2.dilate(image_black, kernel, iterations=1)
    image_black_bottom = image_black[int(height*0.6):]

    # Find contours
    contours, _ = cv2.findContours(image_black_bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour assuming it is the gripper
    # print('Number of contours', len(contours))m
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding box coordinates (x, y, width, height)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate distances
    distance_left = x  # Distance from left boundary to the left side of the gripper
    distance_right = image.shape[1] - (x + w)  # Distance from right boundary to the right side of the gripper
    distance_left /= width
    distance_right /= width

    gripper_state_left = (distance_left - gripper_left[0]) / (gripper_left[1] - gripper_left[0])
    gripper_state_right = (distance_right - gripper_right[0]) / (gripper_right[1] - gripper_right[0])

    return gripper_state_left, gripper_state_right