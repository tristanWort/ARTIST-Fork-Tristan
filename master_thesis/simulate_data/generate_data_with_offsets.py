"""
This is a script for generating data which can used to validate alignment / calibration.

The data which is generated consists of json-files and flux-images.
Based on real sun and motor positions which are extracted from PAINT-data, the simulated
datasets are generated using the ARTIST's Raytracer. 

Before performing Raytracing, the kinematic parameters which were loaded from the scenario file
are altered by adding a random offset which will introduce an alignment error between the 
original scenario's output (original parameters) and the generated output (parameters incl. offsets).

An alignment / caliration algorithm which is given the orignal scenario should be able to find the
new parameters.
"""

import pathlib
import time
import h5py
import torch
import sys
import os
import copy

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.scenario import Scenario
from artist.util import paint_loader, set_logger_config, utils

# Add local path to HeliOptiCal
model_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, model_path) 
from HeliOptiCal.calibration_dataset import CalibrationDataLoader

global_seed = 7
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    "/dss/dsshome1/05/di38kid/data/scenarios/20250325_scenario_7heliostats.h5"
)

# Specify path to paint-data
paint_path = pathlib.Path("/dss/dsshome1/05/di38kid/data/paint")

# Load the scenario.
with h5py.File(scenario_path) as scenario_file:
    loaded_scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )
    
heliostat_field = loaded_scenario.heliostat_field 
loaded_kinematic = heliostat_field.rigid_body_kinematic

print('heliostats in field:')
print(heliostat_field.all_heliostat_names)

# Add random errors to the kinematic parameters.
from add_random_errors import add_random_errors_to_kinematic
# Returns a copy of the kinematic instance with changed paramaters and a dictionary with error values
save_param_dir = r"/dss/dsshome1/05/di38kid/data/results/simulated_data/01/parameters"
modified_kinematic = add_random_errors_to_kinematic(kinematic=loaded_kinematic, 
                                                    save_dir=save_param_dir, 
                                                    heliostat_names=heliostat_field.all_heliostat_names,
                                                    seed=global_seed, 
                                                    device=device)

# For debugging, check if parameter was changed
print(next(iter(loaded_kinematic.all_heliostats_position_params.values()))[0].item())
print(next(iter(modified_kinematic.all_heliostats_position_params.values()))[0].item())


"""For debugging only: Load some example data, later remove."""
# Load some example data
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
all_incident_ray_directions = torch.empty(1, 6, 4).to(device)
all_motor_positions = torch.empty(1, 6, 2).to(device)
all_aim_points = torch.empty(1, 6, 4).to(device)

for h_idx, calibration_paths in enumerate(calibration_paths_dict.values()):
    for s_idx, calibration_path in enumerate(calibration_paths):
        target_name, flux_center, sun_position, motor_positions = paint_loader.extract_paint_calibration_data(
            calibration_properties_path=calibration_path,
            power_plant_position=loaded_scenario.power_plant_position,
            device=device,
        )
    
        target_areas[s_idx].append(loaded_scenario.get_target_area(target_name))
        all_incident_ray_directions[s_idx, h_idx] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_position
        all_motor_positions[s_idx, h_idx]  = motor_positions
        all_aim_points[s_idx, h_idx] = flux_center  

from utils import aim_and_shoot_and_save_bitmaps
# Align the original scenario with the incident ray directions to get proper motor positions
aim_and_shoot_and_save_bitmaps(scenario=loaded_scenario,
                               name='original_scenario',
                               incident_ray_directions=all_incident_ray_directions[0],
                               target_areas=target_areas[0],
                               aim_points=all_aim_points[0],
                               align_with_motor_positions=False,
                               motor_positions=all_motor_positions[0],
                               seed=global_seed,
                               device=device                               
                               )

optimized_motor_positions = loaded_kinematic.motor_positions

# Run raytracer with the modified kinematic
loaded_scenario.heliostat_field.rigid_body_kinematic = modified_kinematic
aim_and_shoot_and_save_bitmaps(scenario=loaded_scenario,
                               name='modified_scenario',
                               incident_ray_directions=all_incident_ray_directions[0],
                               target_areas=target_areas[0],
                               aim_points=all_aim_points[0],
                               align_with_motor_positions=True,
                               motor_positions=optimized_motor_positions,
                               seed=global_seed,
                               device=device                               
                               )

# Setup the calibration DataLoader
all_heliostat_ids = scenario.heliostat_field.all_heliostat_names
calibration_data_loader = CalibrationDataLoader(data_directory=paint_path,
                                                heliostats_to_load=all_heliostat_ids,
                                                power_plant_position=scenario.power_plant_position,
                                                load_flux_images=False,
                                                device=device)

# Generate the k-nn splits
split_config = {
        "path_to_measurements": "/dss/dsshome1/05/di38kid/data/paint/metadata/calibration_metadata_selected_heliostats_20250325_150310.csv",
        "output_dir": "/dss/dsshome1/05/di38kid/data/results",
        "training_sizes": [30],
        "validation_sizes": [30],
        "split_types": ["knn"]
    }                                    
calibration_data_loader.sun_positions_splits(config=split_config, 
                                             save_sun_positions_splits_plots=False)

# Extract first split type and split size from splits
splits = model.calibration_data_loader.splits
_, knn_splits = next(iter(splits.items()))
_, knn_split_df = next(iter(knn_splits.items()))

# Extract heliostat IDs and calibration IDs from Datframe and save in a dictionary
heliostat_and_calib_ids = {heliostat_id: knn_split_df.loc[knn_split_df['HeliostatId'] == heliostat_id].index()
                           for heliostat_id in knn_split_df['HeliostatId'].unique()}     

# Get the calibration data as a batch   
batch = calibration_data_loader.get_field_batch(heliostats_and_calib_ids=heliostats_and_calib_ids)

# Iterate over the batch to perform raytracing and save the results
for n_sample, data in enumerate(batch):
    
    # Set aimpoints to flux center
    kinematic.aim_points = data['flux_centers']
    
    # Use alignment based on incident rays, to make sure that target is not missed.
    kinematic.align_surfaces_with_incident_ray_direction(
        incident_ray_direction=data['incident_rays'],
        device=device
        )
    field_orientations = kinematic.orientations
    
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


scenario.heliostat_field.align_surfaces_with_motor_positions(
   motor_positions=all_motor_positions[0],
   device=device
)

# Create raytracer
raytracer = HeliostatRayTracer(scenario=scenario, world_size=1, rank=0, batch_size=1, random_seed=7)

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
