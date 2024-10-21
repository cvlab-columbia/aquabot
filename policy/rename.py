import time
import os
import shutil
from tqdm import tqdm

def convert_time_to_perf_counter(epoch_time):
    # Capture the current time using both methods
    time_diff = int(time.time()*1000) - int(time.perf_counter()*1000)
    
    # Convert the given epoch time to perf_counter equivalent
    perf_counter_time = epoch_time - time_diff
    
    return perf_counter_time

base_path = '../fifish-vevo/rov_control/data_0806'

for i in tqdm(range(1, 51)):
    episode_path = f'{base_path}/episode_{i}/status'
    new_episode_path = f'{base_path}/episode_{i}/status_new'
    os.makedirs(new_episode_path, exist_ok=True)

    all_files = sorted(os.listdir(episode_path))
    for file in all_files:
        file_path = f'{episode_path}/{file}'
        
        time_time = file.split('.')[0]
        time_time = int(time_time)
        time_perf = convert_time_to_perf_counter(time_time)

        new_file_path = f'{new_episode_path}/{time_perf}.json'
        shutil.copy(file_path, new_file_path)
        
        
