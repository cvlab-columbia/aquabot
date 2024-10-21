import os
import shutil

def find_and_copy_episodes(source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    episode_folders = []
    
    # Recursively search for episode folders
    for root, dirs, files in os.walk(source_dir):
        for folder in dirs:
            if folder.startswith('episode_'):
                episode_folders.append(os.path.join(root, folder))

    # Copy and rename episodes with zero-padded index
    for index, folder in enumerate(sorted(episode_folders), start=1):
        new_folder_name = f"episode_{index:03d}"
        destination_path = os.path.join(dest_dir, new_folder_name)
        shutil.copytree(folder, destination_path)
        print(f"Copied {folder} to {destination_path}")

# Example usage
source_directory = '.'  # Replace with the source directory path
destination_directory = './data'  # Replace with the destination directory path

find_and_copy_episodes(source_directory, destination_directory)

