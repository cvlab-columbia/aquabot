import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os

import sys
sys.path.append('../../policy')

from dataset_sequential import SequentialRoboticDataset
from network import RobotPolicyModel  # Ensure this is implemented
from tqdm import tqdm
import subprocess
import json
import time
from multiprocessing.connection import Client


slices = [0, 1, 2, 3, 4, 5, 6, 13] # non-zero action indices
data_path = 'data_train'
episode_name = 'episode_101'
episode_path = os.path.join(data_path, episode_name)
action_path = os.path.join(episode_path, 'action')
qysea_xbox_address = ('localhost', 8080)

while True:
    try:
        conn = Client(qysea_xbox_address)
        conn.send('ping')
        conn.close()
        break
    except ConnectionRefusedError:
        time.sleep(0.5)

def main():
    all_actions = []
    for action_file in sorted(os.listdir(action_path)):
        action = np.load(os.path.join(action_path, action_file))
        all_actions.append(action)
    
    print('loaded all actions')

    for i in range(len(all_actions)):
        action_str = json.dumps(all_actions[i].tolist())
        with Client(qysea_xbox_address) as conn:
            conn.send(action_str)
        time.sleep(0.05)

    def array_to_string(array):
        return ', '.join([f'{a:.2f}' for a in array])

if __name__ == '__main__':
    main()
