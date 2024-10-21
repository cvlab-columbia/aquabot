import os
import random
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from concurrent.futures import ThreadPoolExecutor
import json

# Paths
object_path = 'object_masks/robot/frames'
background_path = 'pool_images/frames'
train_output_path = 'dataset/train/robot'
test_output_path = 'dataset/test/robot'
train_bbox_output_path = 'dataset/train/robot_with_bbox'
test_bbox_output_path = 'dataset/test/robot_with_bbox'
train_bbox_json_path = 'dataset/train/robot_bboxes.json'
test_bbox_json_path = 'dataset/test/robot_bboxes.json'

# Augmentations
object_augmentations = A.Compose([
    A.ColorJitter(),
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5)
])

background_augmentations = A.Compose([
    A.ColorJitter(),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.HorizontalFlip(p=0.5)
])

noise_augmentations = A.Compose([
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5)
])

# Helper functions
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

def overlay_images(background, overlay, x, y):
    h, w, _ = overlay.shape
    alpha_s = overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (alpha_s * overlay[:, :, c] +
                                       alpha_l * background[y:y+h, x:x+w, c])
    return background

def draw_bounding_box(image, x, y, width, height, color=(0, 255, 0), thickness=2):
    return cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)

def process_image(i, object_files, background_files, output_path, bbox_output_path, bbox_data, bbox_sample_limit, split='train'):
    # Load object image
    obj_img = load_image(random.choice(object_files))
    
    # Separate alpha channel
    alpha_channel = obj_img[:, :, 3]
    obj_img_rgb = obj_img[:, :, :3]

    # Apply random cropping to the object image
    crop_x = random.randint(0, int(obj_img_rgb.shape[1] * 0.3))  # Crop up to 10% from the left
    crop_y = random.randint(0, int(obj_img_rgb.shape[0] * 0.3))  # Crop up to 10% from the top
    crop_w = obj_img_rgb.shape[1] - random.randint(0, int(obj_img_rgb.shape[1] * 0.3))  # Crop up to 10% from the right
    crop_h = obj_img_rgb.shape[0] - random.randint(0, int(obj_img_rgb.shape[0] * 0.3))  # Crop up to 10% from the bottom

    # crop only 50% of the time
    if random.random() > 0.5:
        obj_img_rgb = obj_img_rgb[crop_y:crop_h, crop_x:crop_w]
        alpha_channel = alpha_channel[crop_y:crop_h, crop_x:crop_w]

    # Simulate occlusion by applying a random mask
    mask = np.ones_like(alpha_channel, dtype=np.uint8) * 255
    num_rectangles = random.randint(1, 5)
    for _ in range(num_rectangles):
        x1 = random.randint(0, mask.shape[1] - 1)
        y1 = random.randint(0, mask.shape[0] - 1)
        x2 = random.randint(x1, mask.shape[1])
        y2 = random.randint(y1, mask.shape[0])
        mask[y1:y2, x1:x2] = 0

    alpha_channel = cv2.bitwise_and(alpha_channel, alpha_channel, mask=mask)

    # Load background image and scale to have vertical resolution of 720
    bg_img = load_image(random.choice(background_files))
    bg_img = background_augmentations(image=bg_img)['image']
    scale_factor = 720 / bg_img.shape[0]
    bg_img = cv2.resize(bg_img, (int(bg_img.shape[1] * scale_factor), 720))

    # Resize the object image to fit within the background dimensions
    max_height = min(400, bg_img.shape[0] - 1)
    max_width = min(400, bg_img.shape[1] - 1)

    min_height = 20
    
    # Increase likelihood of smaller objects
    height = random.choices(range(min_height, max_height), weights=[int(i**2) for i in range(max_height-1, min_height-1, -1)])[0]
    width = int(obj_img_rgb.shape[1] * (height / obj_img_rgb.shape[0]))
    if width > max_width:
        width = max_width
        height = int(obj_img_rgb.shape[0] * (width / obj_img_rgb.shape[1]))

    obj_img_rgb = cv2.resize(obj_img_rgb, (width, height))
    alpha_channel = cv2.resize(alpha_channel, (width, height))

    # Apply augmentations
    aug = object_augmentations(image=obj_img_rgb, mask=alpha_channel)
    obj_aug = aug['image']
    alpha_channel_aug = aug['mask']

    # Combine the augmented image and alpha channel back
    obj_aug = cv2.merge((obj_aug, alpha_channel_aug))

    # Random position for the object on the background
    x_offset = random.randint(0, bg_img.shape[1] - obj_aug.shape[1])
    y_offset = random.randint(0, bg_img.shape[0] - obj_aug.shape[0])

    # Overlay images
    result = overlay_images(bg_img.copy(), obj_aug, x_offset, y_offset)

    # Calculate the bounding box as top-left and bottom-right coordinates
    bbox_x1 = x_offset
    bbox_y1 = y_offset
    bbox_x2 = x_offset + obj_aug.shape[1]
    bbox_y2 = y_offset + obj_aug.shape[0]

    # Apply noise augmentation to the final image
    result = noise_augmentations(image=result)['image']

    # Save the result with bounding box overlay option
    result_path = os.path.join(output_path, f'image_{i:06d}.jpg')
    cv2.imwrite(result_path, result, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    if len(bbox_data) < bbox_sample_limit:
        result_with_bbox = draw_bounding_box(result.copy(), bbox_x1, bbox_y1, bbox_x2 - bbox_x1, bbox_y2 - bbox_y1)
        result_bbox_path = os.path.join(bbox_output_path, f'image_{i:06d}_bbox.jpg')
        cv2.imwrite(result_bbox_path, result_with_bbox, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Save bounding box coordinates in pixel values
    bbox_data.append({
        'image': f'image_{i:06d}.jpg',
        'bbox': [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
    })


# Create output directories if not exists
os.makedirs(train_output_path, exist_ok=True)
os.makedirs(test_output_path, exist_ok=True)
os.makedirs(train_bbox_output_path, exist_ok=True)
os.makedirs(test_bbox_output_path, exist_ok=True)

# Get list of files
object_files = [os.path.join(object_path, f) for f in os.listdir(object_path) if f.endswith('.png')]
background_files = [os.path.join(background_path, f) for f in os.listdir(background_path) if f.endswith('.jpg') or f.endswith('.png')]

# Number of threads
num_threads = 72

# Bounding box data
train_bbox_data = []
test_bbox_data = []
bbox_sample_limit = 1000  # Limit for bounding box sample images

# Number of images
num_images = 10000
train_split = 0.96
num_train_images = int(num_images * train_split)
num_test_images = num_images - num_train_images

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    list(tqdm(executor.map(lambda i: process_image(i, object_files, background_files, train_output_path, train_bbox_output_path, train_bbox_data, bbox_sample_limit, 'train'), range(num_train_images)), total=num_train_images))
    list(tqdm(executor.map(lambda i: process_image(i, object_files, background_files, test_output_path, test_bbox_output_path, test_bbox_data, bbox_sample_limit, 'test'), range(num_test_images)), total=num_test_images))

# Save bounding box data to JSON files
with open(train_bbox_json_path, 'w') as f:
    json.dump(train_bbox_data, f, indent=4)

with open(test_bbox_json_path, 'w') as f:
    json.dump(test_bbox_data, f, indent=4)

print("Dataset generation complete.")
