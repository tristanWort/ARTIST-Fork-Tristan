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
    
print(scenario.heliostat_field.all_heliostat_names)

calibration_paths_dict = {
    'AA39': 
        ['/jump/tw/data/paint/AA39/Calibration/158936-calibration-properties.json', 
         '/jump/tw/data/paint/AA39/Calibration/156686-calibration-properties.json'],
    'AM35':
        ['/jump/tw/data/paint/AM35/Calibration/78521-calibration-properties.json',
         '/jump/tw/data/paint/AM35/Calibration/72689-calibration-properties.json']
        }

target_areas = [[],[]]
all_incident_ray_directions = torch.Tensor(2, 2, 4, device=device)
all_motor_positions = torch.Tensor(2, 2, 2, device=device)
all_aim_points = torch.Tensor(2, 2, 4, device=device)

for h_idx, calibration_paths in enumerate(calibration_paths_dict.values()):
    for s_idx, calibration_path in enumerate(calibration_paths):
        target_name, target_center, sun_position, motor_positions = paint_loader.extract_paint_calibration_data(
            calibration_properties_path=calibration_path,
            power_plant_position=scenario.power_plant_position,
            device=device,
        )
    
        target_areas[s_idx].append(scenario.get_target_area(target_name))
        all_incident_ray_directions[s_idx, h_idx] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_position
        all_motor_positions[s_idx, h_idx]  = motor_positions
        all_aim_points[s_idx, h_idx] = target_center

scenario.heliostat_field.rigid_body_kinematic.aim_points = all_aim_points[0]

print('loaded motor positions:', all_motor_positions)
# # Incident ray direction needs to be normed
# incident_ray_direction = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_position
# incident_ray_direction_south = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
#incident_ray_direction = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

# Align all heliostats
scenario.heliostat_field.align_surfaces_with_incident_ray_direction(
    incident_ray_direction=all_incident_ray_directions[0, 1],
    device=device
)

# all_surface_points = scenario.heliostat_field.all_surface_points
# all_surface_normals = scenario.heliostat_field.all_surface_normals

# aligned_surface_points, aligned_surface_normals = scenario.heliostat_field.rigid_body_kinematic.align_surfaces_with_motor_positions(
#     surface_points=all_surface_points, 
#     surface_normals=all_surface_normals, 
#     motor_positions=all_motor_positions, 
#     device=device
# )

scenario.heliostat_field.align_surfaces_with_motor_positions(
    motor_positions=all_motor_positions[0],
    device=device
)

# Create raytracer
raytracer = HeliostatRayTracer(
    scenario=scenario, world_size=world_size, rank=rank, batch_size=1, random_seed=rank
)

# incident_ray_directions = torch.stack((all_incident_ray_directions, all_incident_ray_directions), dim=0)

start_time = time.time()
# Perform heliostat-based raytracing.
final_bitmaps = raytracer.trace_rays_unique_sun(
    incident_ray_directions=all_incident_ray_directions[0],
    target_area=target_areas[0][0],
    device=device
)

end_time = time.time()

#plt.imshow(final_bitmaps[0].cpu().detach(), cmap="inferno")
fig, axs = plt.subplots(nrows=final_bitmaps.shape[0], figsize=(10, 10))

i = 0
j = 0
for i in range(final_bitmaps.shape[0]):
    print(final_bitmaps[i].shape, final_bitmaps[i].max())
    axs[i].imshow(final_bitmaps[i].cpu().detach(), cmap="inferno")

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
