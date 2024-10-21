import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def load_and_sort_images(folder_path):
    # Load image filenames
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Extract timestamps from filenames and sort them
    timestamps = [int(f.split('.')[0]) for f in image_files]
    timestamps.sort()
    
    # Convert timestamps to datetime objects
    datetime_stamps = [datetime.fromtimestamp(ts / 1000.0) for ts in timestamps]
    
    return datetime_stamps

def calculate_time_differences(timestamps):
    # Calculate differences in time between consecutive images
    time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)]
    
    return time_diffs

def plot_histogram(time_diffs):
    print(time_diffs)
    plt.hist(time_diffs, bins=20, edgecolor='black')
    plt.title("Histogram of Time Differences Between Images")
    plt.xlabel("Time Difference (seconds)")
    plt.ylabel("Frequency")
    plt.show()

def main():
    folder_path = 'action'
    timestamps = load_and_sort_images(folder_path)
    
    if len(timestamps) < 2:
        print("Not enough images to calculate time differences.")
        return
    
    time_diffs = calculate_time_differences(timestamps)
    average_time_diff = np.mean(time_diffs) if time_diffs else 0
    print(f"Average time difference between images: {average_time_diff} seconds")
    
    plot_histogram(time_diffs)

if __name__ == "__main__":
    main()
