import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the calibration results
data = np.load('stereo_calib.npz')
R = data['R']
T = data['T']

# Function to create a 3D arrow for camera axes
def plot_camera(ax, R, T, color='b', label='Camera'):
    scale = 50  # Adjust scale for better visualization
    origin = T.flatten()
    x_axis = R[:, 0] * scale
    y_axis = R[:, 1] * scale
    z_axis = R[:, 2] * scale

    ax.quiver(*origin, *x_axis, color=color, length=1, normalize=True, label=f'{label} x-axis')
    ax.quiver(*origin, *y_axis, color=color, length=1, normalize=True, label=f'{label} y-axis')
    ax.quiver(*origin, *z_axis, color=color, length=1, normalize=True, label=f'{label} z-axis')

# Create figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the first camera at the origin
plot_camera(ax, np.eye(3), np.zeros((3, 1)), color='r', label='Camera 1')

# Plot the second camera with the calculated rotation and translation
plot_camera(ax, R, T, color='b', label='Camera 2')

# Set labels
ax.set_xlabel('X axis (mm)')
ax.set_ylabel('Y axis (mm)')
ax.set_zlabel('Z axis (mm)')
ax.set_title('Relative Camera Positions and Orientations')

# Set limits for better visualization
limit = 100  # Adjust the limit based on your T values for better visualization
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])

# Display the plot
plt.legend()
plt.show()
