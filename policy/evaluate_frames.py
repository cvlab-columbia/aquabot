import torch
import cv2
import numpy as np
import os
from network import RobotPolicyModel  # Ensure this is implemented
from tqdm import tqdm

def main():
    n_obs = 2
    n_pred = 2

    # Load the trained model
    model = RobotPolicyModel(n_obs=n_obs, n_pred=n_pred, seperate_encoder=True).cuda()
    model.load_state_dict(torch.load('checkpoints/n_obs_2_n_pred_2_interval_100_batch_size_32_lr_0.0001_loss_mse_seperate_encoder_True_status_conditioned_False__obs_interval_300_bottleneck-dim-None_vae_weight-0.0/checkpoint_epoch_40.pth'))
    model.eval()

    # Specify the folder where frames are saved
    base_folder = '/home/rliu/robotics/zima-blue/fifish-vevo/rov_control/ICRA/qualitative'  # Change this to your base folder path

    # Iterate over each video folder
    video_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    for video_folder in video_folders:
        rov_mount_folder = os.path.join(base_folder, video_folder, 'rov_mount')
        rov_main_folder = os.path.join(base_folder, video_folder, 'rov_main')

        # Load frames from both folders
        rov_mount_frames = sorted([f for f in os.listdir(rov_mount_folder) if f.endswith('.jpg')])
        rov_main_frames = sorted([f for f in os.listdir(rov_main_folder) if f.endswith('.jpg')])

        actions_pred = []

        # Iterate through the frames and generate predictions
        for i in tqdm(range(len(rov_mount_frames) - n_obs)):  # Iterate over available frames
            # Load the observation frames
            obs_mount = []
            obs_main = []

            for j in range(n_obs):
                mount_frame_path = os.path.join(rov_mount_folder, rov_mount_frames[i + j])
                main_frame_path = os.path.join(rov_main_folder, rov_main_frames[i + j])

                mount_frame = cv2.imread(mount_frame_path)
                main_frame = cv2.imread(main_frame_path)

                # Preprocess the images
                mount_frame = cv2.resize(mount_frame, (224, 224))
                main_frame = cv2.resize(main_frame, (224, 224))

                # Normalize the frames as the model expects
                mount_frame = (mount_frame.astype(np.float32) / 255.0) * 2. - 1.
                main_frame = (main_frame.astype(np.float32) / 255.0) * 2. - 1.

                obs_mount.append(mount_frame)
                obs_main.append(main_frame)

            # Convert to tensor format
            obs_mount = torch.tensor(np.array(obs_mount)).permute(0, 3, 1, 2).unsqueeze(0).cuda()
            obs_main = torch.tensor(np.array(obs_main)).permute(0, 3, 1, 2).unsqueeze(0).cuda()

            # Create a batch dictionary similar to the dataloader
            batch = {'rov_mount': obs_mount, 'rov_main': obs_main, \
                     'cctv_left': obs_mount, 'cctv_right': obs_mount} # placeholders, not actually used
            
            # import pdb; pdb.set_trace()

            # Predict the future actions
            with torch.no_grad():
                predicted_action = model(batch).cpu().numpy().squeeze(0)

            # Store the predicted actions
            actions_pred.append(predicted_action[0])

        # Save the predicted actions as npy files in the corresponding folder
        np.save(os.path.join(base_folder, video_folder, 'predicted_actions.npy'), np.array(actions_pred))
        print(f"Saved predicted actions for {video_folder}")

if __name__ == '__main__':
    main()
