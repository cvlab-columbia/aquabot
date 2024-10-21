import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from tqdm import tqdm
import cv2

class Net(nn.Module):
    def __init__(self, prefix='exp-', num_params=4, resume=None):
        super(Net, self).__init__()
        if resume is not None:
            self.dataset = torch.load(resume)
        else:
            self.dataset = []
        '''
        dataset format:
        data[0]: [parameter, reward]
        data[1]: video
        '''
        self.save_dir = './experiments_self-learn'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.video_dir = os.path.join(self.save_dir, 'videos')
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
        self.log_dir = os.path.join(self.save_dir, 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.exp_name = f"{prefix}{datetime.now().strftime('%Y-%m-%d-%H-%M')}.pt"
        self.exp_dir = os.path.join(self.save_dir, self.exp_name)
        self.num_params = num_params

    def forward(self, x):
        return self.model(x)

    def get_random_params(self):
        return np.random.rand(self.num_params)
    
    def resume(self, resume):
        self.dataset = torch.load(resume)

    def update(self, params, fps=10):
        data, video = params

        self.save_video(video, fps=fps) ##TODO: implement save_video
        
        self.dataset.append(data)
        torch.save(self.dataset, self.exp_dir)

    def save_video(self, video, fps=10):
        uid = 'episode_' + str(len(self.dataset)) + '-' + datetime.now().strftime("%Y-%m-%d-%H-%M")
        video_path = f'{self.video_dir}/video-{uid}.mp4'
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (video[0].shape[1], video[0].shape[0]))
        for frame in video:
            out.write(frame)
        out.release()

    def generate_outside_data(self):
        data_outside = []
        for _ in range(100):
            param = torch.rand(self.num_params) * 1.0 - 0.5
            param = torch.where(param > 0, param + 1.0, param)
            param = torch.cat((param, torch.tensor([-10.])))
            data_outside.append(param)
        return data_outside

    def prepare_data(self):
        data_outside = self.generate_outside_data()
        data_all = self.dataset + data_outside
        return torch.stack(data_all)[torch.randperm(len(data_all))]

    def fit(self):
        self.model = nn.Sequential(
            nn.Linear(self.num_params, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 1)
        ).cuda()

        data_valid = self.prepare_data()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=0.1)
        batch_size = max(8, len(data_valid) // 5)
        l1 = nn.L1Loss()

        loss_train_list = []
        print('Training value function...')
        for _ in tqdm(range(200)):
            self.model.train()
            batch = data_valid[np.random.choice(len(data_valid), batch_size, replace=False)]
            x, y = batch[:, :-1].cuda().float(), batch[:, -1].cuda().float()
            prediction = self.model(x)
            loss = l1(prediction.flatten(), y.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train_list.append(loss.item())

        uid = 'episode_' + str(len(self.dataset)) + '-' + datetime.now().strftime("%Y-%m-%d-%H-%M")
        torch.save(self.model, f'{self.log_dir}/model-{uid}.pt')
        torch.save(loss_train_list, f'{self.log_dir}/loss-{uid}.pt')
        print(f'Converged L1 loss: {np.mean(loss_train_list[-10:]):.4f}')

    def get_optimal_params(self):
        x = torch.tensor([0.5] * self.num_params).cuda().requires_grad_(True)
        optimizer = optim.SGD([x], lr=0.01)

        data_valid = self.prepare_data()
        target_distance = data_valid[:, -1].max().item() + 0.5
        l1 = nn.L1Loss()

        print('Optimizing to get optimal params...')
        loss_list = []
        for _ in tqdm(range(50)):
            prediction = self.model(x)
            loss = -prediction
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        uid = 'episode_' + str(len(self.dataset)) + '-' + datetime.now().strftime("%Y-%m-%d-%H-%M")
        torch.save(loss_list, f'{self.log_dir}/inference-{uid}.pt')

        print(f'Optimized design: {x.detach().cpu().numpy()}')
        print(f'Optimized prediction distance: {prediction.item()}')
        return torch.clamp(x, 0, 1).detach().cpu().numpy()
