import pathlib
import time

import h5py
import torch
import sys
import os

from matplotlib import pyplot as plt

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.raytracing import raytracing_utils
from artist.util.scenario import Scenario
from artist.util import paint_loader, set_logger_config, utils

# Add local path to HeliOptiCal
model_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, model_path) 
from HeliOptiCal.util import calculate_intersection

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    "/dss/dsshome1/05/di38kid/data/scenarios/250312-1424_scenario_AM35.h5"
)

# Load the scenario.
with h5py.File(scenario_path) as scenario_file:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

# Also specify the path to your calibration-properties.json file, used only to retrieve realisitc sun position.
calibration_properties_path = pathlib.Path(
   "/dss/dsshome1/05/di38kid/data/simulated_data/21/AM35/60155-simulation-properties.json"
)
target_name, flux_center, sun_position, motor_positions, ideal_flux_center = paint_loader.extract_paint_calibration_data(
        calibration_properties_path=calibration_properties_path,
        power_plant_position=scenario.power_plant_position,
        coord_system='local_enu',
        has_ideal_flux_center=True,
        device=device,
    )

incident_ray_direction = (torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_position).unsqueeze(0)
motor_positions = motor_positions.unsqueeze(0)
flux_center = flux_center.unsqueeze(0)

scenario.heliostat_field.rigid_body_kinematic.aim_points = flux_center

def eval_angles_mrad(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    m1 = torch.norm(v1, dim=-1)
    m2 = torch.norm(v2, dim=-1)
    dot_products = torch.sum(v1 * v2, dim=-1)
    cos_angles = dot_products / (m1 * m2)
    
    angles_rad = torch.acos(torch.clip(cos_angles, -1.0, 1.0))
    
    return angles_rad * 1000

def calc_accuracy_of_alignment(heliostat_orient, incident_ray_dir, target, ideal_flux_c) -> torch.Tensor:
    # Target reflection direction vector from ideal flux center
    target_reflection_direction = (ideal_flux_c - heliostat_orient[0:4, 3])
    
    # Predicted reflection direction vector from momentary orientation of the heliostat 
    reflected_ray = raytracing_utils.reflect(incident_ray_dir, heliostat_orient[0:4, 2])
    flux_center, t = calculate_intersection(
        ray_origin=heliostat_orient[0:4, 3],
        ray_direction=reflected_ray,
        plane_center=target.center,
        plane_normal=target.normal_vector,
    )
    pred_reflection_direction = (flux_center - heliostat_orient[0:4, 3])
    return eval_angles_mrad(pred_reflection_direction, target_reflection_direction)

new_heliostat_e = torch.tensor(-5.565374374389648, device=device)
scenario.heliostat_field.rigid_body_kinematic.all_heliostats_position_params['heliostat_e'][0].data = new_heliostat_e
scenario.heliostat_field.align_surfaces_with_motor_positions(
   motor_positions=motor_positions,
   device=device
)
orientation = scenario.heliostat_field.rigid_body_kinematic.orientations[0]
target_area = scenario.get_target_area(target_name)

acc = calc_accuracy_of_alignment(orientation, incident_ray_direction, target_area, ideal_flux_center)
print(acc.data)

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
        ['/dss/dsshome1/05/di38kid/data/paint/AA39/Calibration/100733-calibration-properties.json'],
    'AC27':
        ['/dss/dsshome1/05/di38kid/data/paint/AC27/Calibration/59587-calibration-properties.json'],
    'AD43':
        ['/dss/dsshome1/05/di38kid/data/paint/AD43/Calibration/65442-calibration-properties.json'],
    'AM35':
        ['/dss/dsshome1/05/di38kid/data/paint/AM35/Calibration/60155-calibration-properties.json'],
    'BB72':
        ['/dss/dsshome1/05/di38kid/data/paint/BB72/Calibration/210898-calibration-properties.json'],
    'BG24':
        ['/dss/dsshome1/05/di38kid/data/paint/BG24/Calibration/171845-calibration-properties.json']
        }

target_areas = [[],[]]
all_incident_ray_directions = torch.Tensor(1, 6, 4, device=device)
all_motor_positions = torch.Tensor(1, 6, 2, device=device)
all_aim_points = torch.Tensor(1, 6, 4, device=device)

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
scenario.heliostat_field.align_surface_with_incident_ray_direction(
    incident_ray_direction=all_incident_ray_directions[0],
    device=device
)

# all_surface_points = scenario.heliostat_field.all_surface_points
# all_surface_normals = scenario.heliostat_field.all_surface_normals

# scenario.heliostat_field.align_surfaces_with_motor_positions(
#    motor_positions=all_motor_positions[0],
#    device=device
# )

# Create raytracer
raytracer = HeliostatRayTracer(
    scenario=scenario, world_size=world_size, rank=rank, batch_size=1, random_seed=rank
)

# incident_ray_directions = torch.stack((all_incident_ray_directions, all_incident_ray_directions), dim=0)

start_time = time.time()
# Perform heliostat-based raytracing.
final_bitmaps = raytracer.trace_rays_separate(
    incident_ray_directions=all_incident_ray_directions[0],
    target_areas=target_areas[0],
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
plt.savefig("/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/tutorials/data/raytracing_sep.png")

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
