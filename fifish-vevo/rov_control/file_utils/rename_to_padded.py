import os

# Get a list of all folders in the current directory
folders = [folder for folder in os.listdir() if os.path.isdir(folder) and folder.startswith('episode_')]

# Sort folders by their numeric part
folders.sort(key=lambda x: int(x.split('_')[1]))

# Rename folders to have zero-padded numbers (3 digits)
for folder in folders:
    # Extract the current episode number
    old_number = int(folder.split('_')[1])
    # Create the new folder name with 3-digit zero-padding
    new_name = f'episode_{old_number:03d}'
    # Rename the folder
    os.rename(folder, new_name)

print("Folders renamed successfully!")

