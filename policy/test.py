from dataset_sequential import SequentialRoboticDataset
import torch
import time
from tqdm import tqdm

dataset = SequentialRoboticDataset(root_dir='../fifish-vevo/rov_control/data_train', n_obs=1, n_pred=8, interval=100, test=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

start_time = time.perf_counter()
for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    pass
print((time.perf_counter() - start_time) / len(dataloader))
