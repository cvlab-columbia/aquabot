import os
import cv2
import torch
import argparse
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np

def load_model(checkpoint_path, device, num_classes=2):
    model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def draw_boxes(image, boxes, scores, color=(255, 0, 0)):
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def main():
    parser = argparse.ArgumentParser(description="Run inference on a video and save the output with bounding boxes.")
    parser.add_argument('--input_video', type=str, required=True, help='Path to the input video file.')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the latest checkpoint
    checkpoint_dir = 'checkpoints'
    checkpoints = sorted([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')], key=os.path.getmtime)
    if not checkpoints:
        print("No checkpoints found.")
        return

    latest_checkpoint = checkpoints[-1]
    model = load_model(latest_checkpoint, device)

    # Open the video file
    video_path = args.input_video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Prepare the frame for inference
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

            # Run inference
            outputs = model(image)
            scores = outputs[0]['scores'].cpu().numpy()
            boxes = outputs[0]['boxes'].cpu().numpy().astype(int)

            # Apply score threshold and get the highest-scored box
            if len(scores) > 0 and np.max(scores) > 0.0:
                best_idx = np.argmax(scores)
                if scores[best_idx] > 0.0:
                    draw_boxes(frame, [boxes[best_idx]], [scores[best_idx]])

            # Write the frame with the drawn boxes
            out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    print("Inference complete and video saved as 'temp.mp4'.")

if __name__ == '__main__':
    main()
