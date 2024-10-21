import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from glob import glob
import random
import torchvision.transforms as transforms
import torchvision.io as io
from concurrent.futures import ThreadPoolExecutor

class RoboticDataset(Dataset):
    def __init__(self, root_dir, n_obs, n_pred, interval, seed=42, test=False):
        self.root_dir = root_dir
        self.n_obs = n_obs
        self.n_pred = n_pred
        self.interval = interval
        self.n_steps = n_obs + n_pred

        # Set the seed for repeatability
        self.seed = seed
        self._set_seed()

        self.episodes = sorted(glob(os.path.join(root_dir, 'episode_*')))
        if test:
            self.episodes = self.episodes[-5:]
        else:
            self.episodes = self.episodes[:-5]

        self.test = test

        self.time_limits = []
        self.data = {}
        
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.rov_mount_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self._preload_data()

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _preload_data(self):
        with ThreadPoolExecutor() as executor:
            futures = []
            for episode in self.episodes:
                futures.append(executor.submit(self._load_episode_data, episode))

            for future in futures:
                episode, ep_data, all_timestamps = future.result()
                self.time_limits.append([all_timestamps.min() + 1000, all_timestamps.max() - 500])  # remove first 1s and last 0.5s of episode
                self.data[episode] = ep_data

    def _load_episode_data(self, episode):
        ep_data = {}
        ep_data['action'] = self._load_files(os.path.join(episode, 'action'), 'npy', load_data=True)
        ep_data['cctv_left'] = self._load_files(os.path.join(episode, 'cctv_left'), 'jpg', load_data=True)
        ep_data['cctv_right'] = self._load_files(os.path.join(episode, 'cctv_right'), 'jpg', load_data=True)
        ep_data['rov_main'] = self._load_files(os.path.join(episode, 'rov_main'), 'jpg', load_data=True)
        ep_data['rov_mount'] = self._load_files(os.path.join(episode, 'rov_mount'), 'jpg', load_data=True)
        ep_data['status_new'] = self._load_files(os.path.join(episode, 'status_new'), 'json', load_data=True)

        all_timestamps = np.concatenate([list(ep_data[k].keys()) for k in ep_data])
        return episode, ep_data, all_timestamps

    def _load_files(self, folder, filetype, load_data=False):
        data = {}
        files = glob(os.path.join(folder, f'*.{filetype}'))
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._load_file, file, filetype): int(os.path.splitext(os.path.basename(file))[0]) for file in files}
            for future in futures:
                timestamp = futures[future]
                data[timestamp] = future.result()
        return data

    def _load_file(self, file, filetype):
        if filetype == 'npy':
            return np.load(file).astype(np.float32)
        elif filetype in ['jpg', 'jpeg', 'png']:
            return io.read_image(file)  # Load image as tensor
        elif filetype == 'json':
            with open(file, 'r') as f:
                return json.load(f)

    def __len__(self):
        if self.test:
            return 1024
        else:
            return 10240

    def __getitem__(self, idx):
        # Randomly select an episode
        episode_idx = random.randint(0, len(self.episodes) - 1)
        episode = self.episodes[episode_idx]
        ep_data = self.data[episode]

        start_time = random.randint(self.time_limits[episode_idx][0], 
                                    self.time_limits[episode_idx][1] - self.n_steps * self.interval)
        timestamps = [start_time + i * self.interval for i in range(self.n_steps)]

        trajectory = {}
        for k in ep_data:
            if k == 'action':
                trajectory[k] = [self._get_closest_data(ep_data[k], t, k) for t in timestamps]
            else:
                trajectory[k] = [self._get_closest_data(ep_data[k], t, k) for t in timestamps[:self.n_obs]]

        # Convert lists of data to tensors
        for k in trajectory:
            if k == 'action':
                trajectory[k] = torch.tensor(np.array(trajectory[k]))
            else:
                trajectory[k] = torch.stack([self._transform_image(data, k) for data in trajectory[k]])

        return trajectory

    def _get_closest_data(self, data_dict, timestamp, data_type):
        timestamps = np.array(list(data_dict.keys()))
        closest_timestamp = timestamps[np.argmin(np.abs(timestamps - timestamp))]
        return data_dict[closest_timestamp]

    def _transform_image(self, data, data_type):
        if data_type in ['cctv_left', 'cctv_right', 'rov_main', 'rov_mount']:
            # Apply different transformations for rov_mount
            if data_type == 'rov_mount':
                return self.rov_mount_transform(data.float() / 255.0)
            else:
                return self.base_transform(data.float() / 255.0)
        elif data_type == 'status_new':
            # Convert dict to tensor by flattening values
            return torch.tensor([data['Compass'],
                                 data['Rov_Attitude_Angle']['Pitch'],
                                 data['Rov_Attitude_Angle']['Roll']]) / 360.  # normalize to [0, 1]

# Usage example:
# dataset = RoboticDataset(root_dir='data', n_obs=5, n_pred=5, interval=2)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
