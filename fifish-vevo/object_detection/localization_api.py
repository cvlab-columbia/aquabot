import os
import cv2
import torch
import argparse
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import skvideo.io
import threading

def triangulate_point(p1, p2, K1, K2, R, t):
    # Convert points to homogeneous coordinates
    p1_hom = np.array([p1[0], p1[1], 1.0])
    p2_hom = np.array([p2[0], p2[1], 1.0])

    # Projection matrix for camera 1
    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    
    # Projection matrix for camera 2
    Rt = np.hstack((R, t.reshape(-1, 1)))
    P2 = np.dot(K2, Rt)
    
    # Triangulate point
    points_4d_hom = cv2.triangulatePoints(P1, P2, p1_hom[:2], p2_hom[:2])
    
    # Convert from homogeneous coordinates to 3D coordinates
    points_3d = points_4d_hom[:3] / points_4d_hom[3]
    
    return points_3d

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

def capture_video_stream(video_reader, stream_id, latest_frames, frame_lock, skip_frames):
    frame_count = 0
    for frame in video_reader:
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with frame_lock:
            latest_frames[stream_id] = frame_bgr

def main():
    parser = argparse.ArgumentParser(description="Run inference on a video stream and visualize the results.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--skip_frames', type=int, default=5, help='Number of frames to skip for real-time performance.')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = load_model(args.checkpoint, device)

    cap1 = skvideo.io.vreader("/dev/video1")
    cap2 = skvideo.io.vreader("/dev/video3")

    latest_frames = [None, None]
    frame_lock = threading.Lock()

    thread1 = threading.Thread(target=capture_video_stream, args=(cap1, 0, latest_frames, frame_lock, args.skip_frames))
    thread2 = threading.Thread(target=capture_video_stream, args=(cap2, 1, latest_frames, frame_lock, args.skip_frames))

    thread1.daemon = True
    thread2.daemon = True

    thread1.start()
    thread2.start()

    screen_width = 1920
    screen_height = 1080

    recording = False
    out = None

    while True:
        with frame_lock:
            if latest_frames[0] is not None and latest_frames[1] is not None:
                frames = []
                for frame in latest_frames:
                    frame = frame.copy()  # Ensure we do not modify the original frame
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = image / 255.0
                    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(image)
                        scores = outputs[0]['scores'].cpu().numpy()
                        boxes = outputs[0]['boxes'].cpu().numpy().astype(int)

                    if len(scores) > 0:
                        best_idx = np.argmax(scores)
                        if scores[best_idx] > 0.05:
                            print(f"Number of scores: {len(scores)}, Highest score: {scores[best_idx]}")
                            draw_boxes(frame, [boxes[best_idx]], [scores[best_idx]])

                    frames.append(frame)

                combined_frame = cv2.hconcat(frames)
                height, width, _ = combined_frame.shape
                scale_factor = min(screen_width / width, screen_height / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                resized_frame = cv2.resize(combined_frame, (new_width, new_height))
                cv2.imshow('Video Stream', resized_frame)

                if recording:
                    if out is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter('stream_detection.mp4', fourcc, 24.0, (new_width, new_height))
                    out.write(resized_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            recording = not recording
            if not recording and out is not None:
                out.release()
                out = None

    if out is not None:
        out.release()

    cv2.destroyAllWindows()

    thread1.join()
    thread2.join()

if __name__ == '__main__':
    main()
