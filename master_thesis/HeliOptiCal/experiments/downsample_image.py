import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Settings
size = 20  # Number of pixels in each dimension
sigma = 3  # Controls Gaussian spread

# Generate 2D grid of (x, y) coordinates
x = np.linspace(0, size - 1, size)
y = np.linspace(0, size - 1, size)
xv, yv = np.meshgrid(x, y)
center = (size - 1) / 2

# 2D Gaussian: higher in the center, fading outwards
gaussian = np.exp(-((xv - center) ** 2 + (yv - center) ** 2) / (2 * sigma ** 2))
gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())  # Normalize to 0-1

# Save the raw image
plt.imsave('/dss/dsshome1/05/di38kid/data/results/plots/blocking_function/gaussian_blob.png', gaussian, cmap='gray', vmin=0, vmax=1)

# Now plot with visible pixel grid
fig, ax = plt.subplots(figsize=(6, 6))
cmap = plt.cm.gray
bounds = np.linspace(0, 1, 257)  # for precise mapping
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

im = ax.imshow(gaussian, cmap=cmap, interpolation='none', norm=norm)

# Draw grid lines (for every pixel)
ax.set_xticks(np.arange(-.5, size, 1), minor=True)
ax.set_yticks(np.arange(-.5, size, 1), minor=True)
ax.grid(which='minor', color='black', linewidth=0.7)

# Remove axis labels and ticks
ax.set_xticks([])
ax.set_yticks([])
ax.tick_params(left=False, bottom=False)

plt.tight_layout()
plt.savefig('/dss/dsshome1/05/di38kid/data/results/plots/blocking_function/gaussian_blob_with_grid.png', dpi=300)
plt.show()
