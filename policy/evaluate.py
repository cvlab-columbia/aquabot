import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
from dataset_sequential import SequentialRoboticDataset
from network import RobotPolicyModel  # Ensure this is implemented
from tqdm import tqdm

slices = [0, 1, 2, 3, 4, 5, 6] # non-zero action indices

def main():
    n_obs = 4
    # Load the testing dataset
    test_dataset = SequentialRoboticDataset(root_dir='../fifish-vevo/rov_control/data_train', 
                                            n_obs=n_obs, n_pred=1, interval=100, test=True, obs_interval=200)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Load the trained model
    model = RobotPolicyModel(n_obs=n_obs, n_pred=1, seperate_encoder=True).cuda()
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('checkpoints/n_obs_2_n_pred_1_interval_100_batch_size_128_lr_0.001_loss_l1_seperate_encoder_True_status_conditioned_False_200_resnet/best_model.pth'))
    model.eval()

    # Create directory to save the output video
    save_dir = 'output_videos'
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, 'combined_video_l1.mp4')

    # Prepare the video writer
    height, width = 360, 640  # Assuming all images are resized to 224x224
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))

    # Iterate through the test dataset
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            batch = {k: v.cuda() for k, v in batch.items()}

            # Truncate observation steps as input
            actual_action = batch['action'][:, n_obs:]
            batch['action'] = batch['action'][:, :n_obs]

            # Predict the future actions
            predicted_action = model(batch)

            # Convert the tensor predictions and actuals to numpy
            predicted_action = predicted_action.cpu().numpy().squeeze(0)
            predicted_action = predicted_action[:, slices]
            actual_action = actual_action.cpu().numpy().squeeze(0)
            actual_action = actual_action[:, slices]

            # Use the last frame from the observation frames
            frame = batch['rov_mount'][0, -1].cpu().numpy().transpose(1, 2, 0)
            frame = (frame + 1) / 2  # Denormalize the image
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = (frame * 255).astype(np.uint8)

            # Plot the first 3 predicted and actual actions on the frame
            # import pdb; pdb.set_trace()
            t = 0
            pred_text = "Pred {}: ".format(t+1) + ", ".join(f"{p:.2f}" for p in predicted_action[t])
            actual_text = "Actual {}: ".format(t+1) + ", ".join(f"{a:.2f}" for a in actual_action[t])
            cv2.putText(frame, pred_text, (10, 30 + t * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, actual_text, (10, 50 + t * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Write the frame to the video
            out.write(frame)

    # Release the video writer
    out.release()
    print(f"Saved video: {video_path}")

    def array_to_string(array):
        return ', '.join([f'{a:.2f}' for a in array])

if __name__ == '__main__':
    main()
