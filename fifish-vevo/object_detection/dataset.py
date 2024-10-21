import os
import json
import torch
from PIL import Image
import torchvision.transforms as T

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, bbox_json, transforms=None):
        self.root = root
        self.transforms = transforms
        self.bbox_data = json.load(open(bbox_json))
        self.imgs = list(sorted(os.listdir(root)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        bbox_info = next(item for item in self.bbox_data if item["image"] == self.imgs[idx])
        boxes = [bbox_info['bbox']]  # list of boxes
        
        # Convert boxes to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Ensure that the bounding boxes are valid
        valid_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            if xmax > xmin and ymax > ymin:
                valid_boxes.append([xmin, ymin, xmax, ymax])
        
        if len(valid_boxes) == 0:
            # If no valid boxes, add a dummy box
            print(f"Invalid box for image {self.imgs[idx]}")
            valid_boxes.append([0, 0, 1, 1])

        boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)

        labels = torch.ones((len(boxes),), dtype=torch.int64)  # all objects are of class 1
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

# Example usage:
# train_dataset = CustomDataset(root='dataset/train/robot', bbox_json='dataset/train/robot_bboxes.json', transforms=get_transform(train=True))
# test_dataset = CustomDataset(root='dataset/test/robot', bbox_json='dataset/test/robot_bboxes.json', transforms=get_transform(train=False))
