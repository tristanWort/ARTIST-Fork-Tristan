import pathlib
import time

import h5py
import torch

import sys
import os

from matplotlib import pyplot as plt

repo_path = os.path.abspath(os.path.dirname('/jump/tw/master_thesis/ARTIST-Fork-Tristan/artist'))  # Get the repo directory
sys.path.insert(0, repo_path) 
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.scenario import Scenario
from artist.util import paint_loader, set_logger_config, utils

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

scenario_path = pathlib.Path(r"/jump/tw/data/paint/_h5_scenario_files")
# Specify the path to your scenario.h5 file.
path_file = scenario_path / r"250306-0952_scenario_AA39-AM35.h5"

# The distributed environment is setup and destroyed using a Generator object.
environment_generator = utils.setup_distributed_environment(device=device)

device, is_distributed, rank, world_size = next(environment_generator)

# Load the scenario.
with h5py.File(path_file) as scenario_file:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

# Dict to store target names and their corresponding target areas.
targets_dict = {target.name: target for target in scenario.target_areas.target_area_list}

print("Init aim points:", scenario.heliostat_field.all_aim_points)
new_aim_points = torch.tensor(data=[
    targets_dict['multi_focus_tower'].center.tolist(),
    targets_dict['receiver'].center.tolist()
], device=device)

# Set aim_points for all heliostats.
scenario.heliostat_field.all_aim_points = new_aim_points
print("New aim points:", scenario.heliostat_field.all_aim_points)

# Use south sun position
incident_ray_direction_south = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
# Align all heliostats
scenario.heliostat_field.align_surfaces_with_incident_ray_direction(
    incident_ray_direction=incident_ray_direction_south,
    device=device
)

# Create raytracer
raytracer = HeliostatRayTracer(
    scenario=scenario, world_size=world_size, rank=rank, batch_size=4, random_seed=rank
)
# Perform heliostat-based raytracing.
final_bitmap = raytracer.trace_rays(
    incident_ray_direction=incident_ray_direction_south,
    target_area=targets_dict['multi_focus_tower'],
    device=device
)
print(final_bitmap.max())

plt.imshow(final_bitmap.cpu().detach(), cmap="inferno")
plt.title(f"Flux Density Distribution from rank (heliostat): {rank}")
plt.savefig(f"AA_new_rank_{rank}_{device.type}.png")

final_bitmap = raytracer.normalize_bitmap(final_bitmap,
                                          targets_dict['multi_focus_tower']
                                          )

plt.imshow(final_bitmap.cpu().detach(), cmap="inferno")
plt.title("Total Flux Density Distribution")
plt.savefig(f"AA_new_final_single_device_mode_{device.type}.png")

# Make sure the code after the yield statement in the environment Generator
# is called, to clean up the distributed process group.
try:
    next(environment_generator)
except StopIteration:
    pass
