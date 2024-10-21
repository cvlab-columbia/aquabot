import os

# Get a list of all folders in the current directory
folders = [folder for folder in os.listdir() if os.path.isdir(folder) and folder.startswith('episode_')]

# Sort folders by their numeric part in descending order
folders.sort(key=lambda x: int(x.split('_')[1]), reverse=True)

# Rename folders
for folder in folders:
    # Extract the current episode number
    old_number = int(folder.split('_')[1])
    # Calculate the new episode number
    new_number = old_number + 3
    # Create the new folder name
    new_name = f'episode_{new_number:03d}'
    # Rename the folder
    os.rename(folder, new_name)

print("Folders renamed successfully!")

