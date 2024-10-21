import os
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import DataLoader
import cv2
from dataset import CustomDataset, get_transform
from tqdm import tqdm
from tqdm import tqdm

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        if i % print_freq == 0:
            loss_details = ' | '.join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
            print(f"Epoch [{epoch+1}], Iteration [{i}], Loss: {losses.item():.4f}, Details: {loss_details}")


def calculate_iou(box1, box2):
    # Calculate the Intersection over Union (IoU) of two bounding boxes.
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2g - x1g + 1) * (y2g - y1g + 1)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

def evaluate(model, data_loader, device, epoch):
    model.eval()
    results_dir = f'results/epoch_{epoch}'
    os.makedirs(results_dir, exist_ok=True)
    
    total_iou = 0.0
    num_images = 0
    save_images = True
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for j in range(len(images)):
                img = images[j].cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                gt_box = targets[j]['boxes'].cpu().numpy().astype(int)[0]

                if outputs[j]['boxes'].size(0) > 0:
                    pred_box = outputs[j]['boxes'].cpu().numpy().astype(int)[0]
                    iou = calculate_iou(gt_box, pred_box)
                    total_iou += iou

                    cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 2)
                    cv2.rectangle(img, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (255, 0, 0), 2)
                else:
                    # No detections, IoU is 0
                    cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 2)
                    total_iou += 0

                if save_images:
                    cv2.imwrite(os.path.join(results_dir, f'image_{num_images:06d}.jpg'), img)

                num_images += 1

            if save_images:
                save_images = False  # Only save images from the first batch

    mean_iou = total_iou / num_images
    print(f"Epoch [{epoch+1}] - Mean IoU: {mean_iou:.4f}")


    mean_iou = total_iou / num_images
    print(f"Epoch [{epoch+1}] - Mean IoU: {mean_iou:.4f}")

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2  # 1 class (object) + background
    batch_size = 128  # Updated batch size
    num_epochs = 10
    learning_rate = 0.0001  # Learning rate for AdamW
    weight_decay = 0.0005

    train_dataset = CustomDataset(root='dataset/train/robot', bbox_json='dataset/train/robot_bboxes.json', transforms=get_transform(train=True))
    test_dataset = CustomDataset(root='dataset/test/robot', bbox_json='dataset/test/robot_bboxes.json', transforms=get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

    model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)  # Updated to AdamW optimizer

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        evaluate(model, test_loader, device, epoch)
        torch.save(model.state_dict(), f'checkpoints/checkpoint_{epoch + 1}.pth')

    print("Training complete.")


if __name__ == '__main__':
    main()
