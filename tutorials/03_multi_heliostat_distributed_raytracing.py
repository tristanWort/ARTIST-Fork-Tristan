import pathlib
import time

import h5py
import torch
import sys
import os

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/jump/tw/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.scenario import Scenario
from artist.util import paint_loader, set_logger_config, utils

from matplotlib import pyplot as plt

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    "/jump/tw/data/paint/_h5_scenario_files/250306-0952_scenario_AA39-AM35.h5"
)

# # Also specify the path to your calibration-properties.json file, used only to retrieve realisitc sun position.
# calibration_properties_path = pathlib.Path(
#     "/workVERLEIHNIX/mb/ARTIST/tutorials/data/paint/AA39/calibration-properties.json"
# )

# The distributed environment is setup and destroyed using a Generator object.
environment_generator = utils.setup_distributed_environment(device=device)

device, is_distributed, rank, world_size = next(environment_generator)

# Load the scenario.
with h5py.File(scenario_path) as scenario_file:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

# # Load the incident_ray_direction from the calibration data.
# (
#     _,
#     _,
#     sun_position,
#     _,
# ) = paint_loader.extract_paint_calibration_data(
#     calibration_properties_path=calibration_properties_path,
#     power_plant_position=scenario.power_plant_position,
#     device=device,
# )

# # Incident ray direction needs to be normed
# incident_ray_direction = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_position
incident_ray_direction_south = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
#incident_ray_direction = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

# Align all heliostats
scenario.heliostat_field.align_surfaces_with_incident_ray_direction(
    incident_ray_direction=incident_ray_direction_south,
    device=device
)


# Create raytracer
raytracer = HeliostatRayTracer(
    scenario=scenario, world_size=world_size, rank=rank, batch_size=2, random_seed=rank
)

start_time = time.time()
# Perform heliostat-based raytracing.
final_bitmaps = raytracer.trace_rays(
    incident_ray_direction=incident_ray_direction_south,
    target_area=scenario.get_target_area("receiver"),
    device=device
)

end_time = time.time()

fig, axs = plt.subplots(1, final_bitmaps.shape[0], figsize=(10, 5))

i = 0
while i < final_bitmaps.shape[0]:
    axs[i].imshow(final_bitmaps[i].cpu().detach(), cmap="inferno")
    i += 1
    
plt.title(f"Flux Density Distributions")
plt.savefig("/jump/tw/master_thesis/ARTIST-Fork-Tristan/tutorials/data/raytracing/raytracing_sep.png")

if is_distributed:
    torch.distributed.all_reduce(final_bitmaps, op=torch.distributed.ReduceOp.SUM)

#final_bitmap = raytracer.normalize_bitmap(final_bitmap, aimpoint_area)


print(f"{device}, time: {end_time-start_time}")

# plt.imshow(final_bitmap.cpu().detach(), cmap="inferno")
# plt.title("Total Flux Density Distribution")
# plt.savefig(f"AA_new_final_single_device_mode_{device.type}.png")

# Make sure the code after the yield statement in the environment Generator
# is called, to clean up the distributed process group.
try:
    next(environment_generator)
except StopIteration:
    pass
