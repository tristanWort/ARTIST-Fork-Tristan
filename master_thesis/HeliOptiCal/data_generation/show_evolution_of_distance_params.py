import torch
import sys
import os
import matplotlib.pyplot as plt

# Add local path to HeliOptiCal
model_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, model_path) 
from HeliOptiCal.utils.util_simulate import apply_artificial_blocking, distance_to_blocking_strength


# Generate distances from 0 to 1000 m
distances = torch.linspace(0, 1000, steps=1000)
strengths = torch.tensor([max(1- d.item() / 500, 0.5) for d in distances])

# strengths = distance_to_blocking_strength(distances)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(distances.numpy(), strengths.numpy(), label="Blocking Strength", color='royalblue')
plt.xlabel("Distance to Target [m]")
plt.ylabel("Blocking Strength")
plt.title("Mapping Heliostat Distance -> Blocking Strength")
plt.grid(True, linestyle="--", alpha=0.6)
plt.axhline(0.5, color='gray', linestyle='--', label="Min Blocking Strength = 0.5")
plt.legend()
plt.tight_layout()
# Save the figure
# plt.savefig("/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/data_generation/blocking_strength_vs_distance.png", dpi=300)


# Compute sigma values according to your formula
sigmas = torch.tensor([min(d.item() / 12.5, 40) for d in distances])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(distances.numpy(), sigmas.numpy(), color='darkgreen', label='sigma(distance) = min(distance / 12.5, 40)')
plt.xlabel("Distance to Target [m]")
plt.ylabel("Sigma Value")
plt.title("Mapping Heliostat Distance -> Edge Softening")
plt.grid(True, linestyle="--", alpha=0.6)
plt.axhline(40, color='gray', linestyle='--', label="Max Sigma = 40")
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig("/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/data_generation/sigma_vs_distance.png", dpi=300)
