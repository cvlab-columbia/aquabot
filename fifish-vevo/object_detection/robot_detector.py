import os
import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class RobotDetector:
    def __init__(self, checkpoint_path, calibration_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(checkpoint_path)
        self.K1, self.K2, self.R, self.t = self.load_calibration_matrices(calibration_path)
        self.front, self.origin, self.right, self.up = self.load_vectors()

    def load_model(self, checkpoint_path):
        model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def load_calibration_matrices(self, calibration_path):
        K1 = np.load(os.path.join(calibration_path, 'intrinsic_matrix_left.npy'))
        K2 = np.load(os.path.join(calibration_path, 'intrinsic_matrix_right.npy'))
        R = np.load(os.path.join(calibration_path, 'rotation_matrix.npy'))
        t = np.load(os.path.join(calibration_path, 'translation_vector.npy'))[:, 0]
        return K1, K2, R, t

    def load_vectors(self):
        front = np.load(os.path.join('../object_detection', 'front.npy'))
        origin = np.load(os.path.join('../object_detection', 'origin.npy'))
        right = np.load(os.path.join('../object_detection', 'right.npy'))
        up = np.load(os.path.join('../object_detection', 'up.npy'))
        return front, origin, right, up

    def triangulate_point(self, p1, p2):
        # Convert points to homogeneous coordinates
        p1_hom = np.array([p1[0], p1[1], 1.0])
        p2_hom = np.array([p2[0], p2[1], 1.0])

        # Projection matrix for camera 1
        P1 = np.dot(self.K1, np.hstack((np.eye(3), np.zeros((3, 1)))))

        # Projection matrix for camera 2
        Rt = np.hstack((self.R, self.t.reshape(-1, 1)))
        P2 = np.dot(self.K2, Rt)

        # Triangulate point
        points_4d_hom = cv2.triangulatePoints(P1, P2, p1_hom[:2], p2_hom[:2])

        # Convert from homogeneous coordinates to 3D coordinates
        points_3d = points_4d_hom[:3] / points_4d_hom[3]

        return points_3d

    def project_to_coordinate_frame(self, point):
        # Define the coordinate frame axes
        axis_right = self.right - self.origin
        axis_front = self.front - self.origin
        axis_up = self.up - self.origin

        # Normalize the axes
        axis_right_norm = axis_right / np.linalg.norm(axis_right)
        axis_front_norm = axis_front / np.linalg.norm(axis_front)
        axis_up_norm = axis_up / np.linalg.norm(axis_up)

        # Project the point onto each axis
        projection_right = np.dot(point - self.origin, axis_right_norm)
        projection_front = np.dot(point - self.origin, axis_front_norm)
        projection_up = np.dot(point - self.origin, axis_up_norm)

        return projection_right, projection_front, projection_up
    
    def get_brightest_region(self, image, min_contour_area=100):
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to smooth the image and make bright regions more pronounced
        blurred = cv2.GaussianBlur(gray_image, (15, 15), 0)
        
        # Find the maximum brightness value in the blurred image
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(blurred)
        
        # Threshold the image to isolate the bright regions
        _, thresh = cv2.threshold(blurred, maxVal - 10, 255, cv2.THRESH_BINARY)
        
        # Find contours of the bright regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by size
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
        
        if not filtered_contours:
            return None
        
        # Sort the remaining contours by their topmost point (y-coordinate), smallest first
        filtered_contours = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[1])
        
        # Take the most upper contour
        most_upper_contour = filtered_contours[0]
        
        # Fit a minimum enclosing circle to the most upper contour
        ((x, y), radius) = cv2.minEnclosingCircle(most_upper_contour)
        
        # Return the center of this circle
        return (int(x), int(y))

    

    def detect_robot_3d(self, image_left, image_right, save_detection=False, night=False, return_2d=False):
        # Prepare images for the model
        image_left_tensor = self.preprocess_image(image_left)
        image_right_tensor = self.preprocess_image(image_right)

        if not night:
            # Detect robot in both images
            with torch.no_grad():
                outputs_left = self.model(image_left_tensor)
                outputs_right = self.model(image_right_tensor)

            # Get the highest scoring boxes
            p1 = self.get_best_detection(outputs_left)
            p2 = self.get_best_detection(outputs_right)

            if p1 is None or p2 is None:
                raise ValueError("Could not detect robot in one or both images.")
        
        else:
            # Detect the brightest region in both images
            p1 = self.get_brightest_region(image_left)
            p2 = self.get_brightest_region(image_right)

            if p1 is None or p2 is None:
                raise ValueError("Could not detect the brightest region in one or both images.")


        # Plot detection centers on the images and save them if requested
        if save_detection:
            image_left_detected = image_left.copy()
            image_right_detected = image_right.copy()

            cv2.circle(image_left_detected, tuple(p1), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.circle(image_right_detected, tuple(p2), radius=5, color=(0, 255, 0), thickness=-1)

            cv2.imwrite('left.jpg', image_left_detected)
            cv2.imwrite('right.jpg', image_right_detected)

        # Triangulate the 3D point
        points_3d = self.triangulate_point(p1, p2)

        # Project to the world coordinate frame
        world_coordinates = self.project_to_coordinate_frame(points_3d[:, 0])

        if return_2d:
            return list(world_coordinates), p1, p2

        return list(world_coordinates)


    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return image

    def get_best_detection(self, outputs):
        scores = outputs[0]['scores'].cpu().numpy()
        boxes = outputs[0]['boxes'].cpu().numpy().astype(int)

        if len(scores) > 0:
            best_idx = np.argmax(scores)
            if scores[best_idx] > 0.05:
                center = [int((boxes[best_idx][0] + boxes[best_idx][2]) / 2), int((boxes[best_idx][1] + boxes[best_idx][3]) / 2)]
                return center
        return None

if __name__ == '__main__':
    detector = RobotDetector(checkpoint_path='../object_detection/checkpoints/checkpoint_10.pth', calibration_path='../camera_control/calibration_res_07-30')
    image_left = cv2.imread('../rov_control/data_train/episode_001/cctv_left/22246571.jpg')
    image_right = cv2.imread('../rov_control/data_train/episode_001/cctv_right/22246572.jpg')
    import pdb; pdb.set_trace
    robot_3d_coords = detector.detect_robot_3d(image_left, image_right, save_detection=True, night=True)
    print(f"Robot 3D Coordinates: {robot_3d_coords}")
