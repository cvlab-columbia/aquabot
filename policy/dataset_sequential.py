import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import sys

img_size = 224

class SequentialRoboticDataset(Dataset):
    def __init__(self, root_dir, n_obs, n_pred, interval, obs_interval=100, test=False, image_size = None):
        self.root_dir = root_dir
        self.n_obs = n_obs
        self.n_pred = n_pred
        self.interval = interval
        self.obs_interval = obs_interval
        self.n_steps = n_obs + n_pred
        if image_size is not None:
            self.img_size = image_size
        else:
            self.img_size = [img_size, img_size]

        self.episodes = sorted(glob(os.path.join(root_dir, 'episode_*')))

        # pick every 15th episode for testing
        test_episodes = list(range(0, len(self.episodes), 20))
        if test:
            self.episodes = [ep for i, ep in enumerate(self.episodes) if i in test_episodes]
        else:
            self.episodes = [ep for i, ep in enumerate(self.episodes) if i not in test_episodes]
            
        self.test = test
        self.data = []
        self.start_times = []

        for episode in tqdm(self.episodes):
            ep_data = {}
            ep_data['action'] = self._load_files(os.path.join(episode, 'action'), 'npy')
            ep_data['cctv_left'] = self._load_files(os.path.join(episode, 'cctv_left'), 'jpg')
            ep_data['cctv_right'] = self._load_files(os.path.join(episode, 'cctv_right'), 'jpg')
            ep_data['rov_main'] = self._load_files(os.path.join(episode, 'rov_main'), 'jpg')
            ep_data['rov_mount'] = self._load_files(os.path.join(episode, 'rov_mount'), 'jpg')
            # ep_data['status'] = self._load_files(os.path.join(episode, 'status'), 'json')

            all_timestamps = np.concatenate([list(ep_data[k].keys()) for k in ep_data])
            min_time = int(all_timestamps.min() + 200)
            max_time = int(all_timestamps.max() - 200)
            
            episode_length = self.n_obs * self.obs_interval + self.n_pred * self.interval
            episode_start_times = list(range(min_time, max_time - episode_length, self.interval))
            self.start_times.append(episode_start_times)
            self.data.append(ep_data)
        
        self.base_transform = A.Compose([
            A.Resize(self.img_size[0], self.img_size[1]),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.1, p=0.5),
            A.ToGray(p=0.2),  # Convert to grayscale with a probability of 20%
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Add Gaussian noise with a variance between 10 and 50
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

        self.rov_mount_transform = A.Compose([
            A.Rotate(limit=10),
            A.RandomResizedCrop(self.img_size[0], self.img_size[1], scale=(0.8, 1.0), ratio=(0.95, 1.05)),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.1, p=0.5),
            A.ToGray(p=0.2),  # Convert to grayscale with a probability of 20%
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Add Gaussian noise with a variance between 10 and 50
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

        self.test_transform = A.Compose([
            A.Resize(self.img_size[0], self.img_size[1]),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def _load_files(self, folder, filetype):
        data = {}
        files = glob(os.path.join(folder, f'*.{filetype}'))
        for file in files:
            timestamp = int(os.path.splitext(os.path.basename(file))[0])
            data[timestamp] = file  # Store the file path
        return data

    def __len__(self):
        return sum(len(start_times) for start_times in self.start_times)

    def __getitem__(self, idx):
        episode_idx, local_idx = self._get_episode_and_local_idx(idx)
        start_time = self.start_times[episode_idx][local_idx]
        ep_data = self.data[episode_idx]
        
        timestamps = []
        for i in range(self.n_obs):
            timestamps.append(start_time + i * self.obs_interval)
        for i in range(self.n_pred):
            timestamps.append(start_time + self.n_obs * self.obs_interval + i * self.interval)

        trajectory = {}
        for k in ep_data:
            if k == 'action':
                trajectory[k] = [self._get_closest_data(ep_data[k], t, k) for t in timestamps]
            else:
                trajectory[k] = [self._get_closest_data(ep_data[k], t, k) for t in timestamps[:self.n_obs]]

        for k in trajectory:
            if k == 'action':
                trajectory[k] = torch.tensor(np.array(trajectory[k]))
            else:
                trajectory[k] = torch.stack([self._to_tensor(data, k) for data in trajectory[k]])

        return trajectory

    def _get_episode_and_local_idx(self, idx):
        running_count = 0
        for episode_idx, episode_start_times in enumerate(self.start_times):
            if idx < running_count + len(episode_start_times):
                local_idx = idx - running_count
                return episode_idx, local_idx
            running_count += len(episode_start_times)
        raise IndexError("Index out of range")

    def _get_closest_data(self, data_dict, timestamp, data_type):
        timestamps = np.array(list(data_dict.keys()))
        try:
            closest_timestamp = timestamps[np.argmin(np.abs(timestamps - timestamp))]
        except:
            print(data_type)
            print(timestamp)
            sys.exit()
        file_path = data_dict[closest_timestamp]

        if data_type == 'action':
            try:
                return np.load(file_path).astype(np.float32)
            except:
                print(file_path)
                sys.exit()
        elif data_type in ['cctv_left', 'cctv_right', 'rov_main', 'rov_mount']:
            return Image.open(file_path)
        elif data_type == 'status':
            with open(file_path, 'r') as f:
                return json.load(f)

    def _to_tensor(self, data, data_type):
        if data_type in ['cctv_left', 'cctv_right', 'rov_main', 'rov_mount']:
            if not self.test:
                if data_type == 'rov_mount':
                    augmented = self.rov_mount_transform(image=np.array(data))
                else:
                    augmented = self.base_transform(image=np.array(data))
                return augmented['image']
            else:
                augmented = self.test_transform(image=np.array(data))
                return augmented['image']
        elif data_type == 'status':
            return torch.tensor([data['Compass'],
                                 data['Rov_Attitude_Angle']['Pitch'],
                                 data['Rov_Attitude_Angle']['Roll']]) / 360.  # normalize to [0, 1]
