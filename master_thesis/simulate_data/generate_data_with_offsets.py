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
import json

from pathlib import Path
from matplotlib import pyplot as plt

from utils import aim_and_shoot_and_save_bitmaps, align_and_raytrace

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

global_seed = 42
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

# Specify the path for saving the generated data
save_dir = Path("/dss/dsshome1/05/di38kid/data/simulated_data/02")

# Specify path to paint-data
paint_path = pathlib.Path("/dss/dsshome1/05/di38kid/data/paint")

# Load two scenarios.
with h5py.File(scenario_path) as scenario_file:
    loaded_scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )
    # modified_scenario = Scenario.load_scenario_from_hdf5(
    #     scenario_file=scenario_file, device=device
    # )

heliostat_field = loaded_scenario.heliostat_field 
loaded_kinematic = heliostat_field.rigid_body_kinematic

# Get the list of all heliostats in the field
all_heliostat_ids = loaded_scenario.heliostat_field.all_heliostat_names
print('heliostats in field:')
print(all_heliostat_ids)

# Setup the calibration DataLoader
calibration_data_loader = CalibrationDataLoader(data_directory=paint_path,
                                                heliostats_to_load=all_heliostat_ids,
                                                power_plant_position=loaded_scenario.power_plant_position,
                                                load_flux_images=False,
                                                device=device)

# Generate the k-nn splits
split_config = {
        "path_to_measurements": "/dss/dsshome1/05/di38kid/data/paint/metadata/calibration_metadata_selected_heliostats_20250325_150310.csv",
        "output_dir": save_dir / 'splits',
        "training_sizes": [30],
        "validation_sizes": [30],
        "split_types": ["knn"]
    }                                    
calibration_data_loader.sun_positions_splits(config=split_config, 
                                             save_sun_positions_splits_plots=False)


"""Modify the scenario by adding random offsets to its parameters in the kinematic instance."""
# Add random errors to the kinematic parameters.
from add_random_errors import add_random_errors_to_kinematic

# Returns a copy of the kinematic instance with changed paramaters
modified_kinematic = add_random_errors_to_kinematic(kinematic=loaded_kinematic, 
                                                    save_dir=save_dir / 'parameters', 
                                                    heliostat_names=heliostat_field.all_heliostat_names,
                                                    seed=global_seed, 
                                                    device=device)

# Replace the 'old' kinematic with the modified kinematic
modified_scenario = copy.deepcopy(loaded_scenario)
modified_scenario.target_areas = loaded_scenario.target_areas
modified_scenario.heliostat_field.rigid_body_kinematic = modified_kinematic

# For debugging, check if parameter was changed
print(next(iter(loaded_kinematic.all_heliostats_position_params.values()))[0].item())
print(next(iter(modified_kinematic.all_heliostats_position_params.values()))[0].item())

# Extract first split type and split size from splits
splits = calibration_data_loader.splits
_, knn_splits = next(iter(splits.items()))
_, knn_split_df = next(iter(knn_splits.items()))

# Extract heliostat IDs and calibration IDs from Datframe and save in a dictionary
heliostat_and_calib_ids = {heliostat_id: knn_split_df.loc[knn_split_df['HeliostatId'] == heliostat_id].index.tolist()
                           for heliostat_id in knn_split_df['HeliostatId'].unique()}     

# Get the calibration data as a batch   
batch = calibration_data_loader.get_field_batch(heliostats_and_calib_ids=heliostat_and_calib_ids)

properties_dict = {
    "motor_position": {"axis_1_motor_position": 0,
                       "axis_2_motor_position": 0},
    "surface_normal": [],
    "target_name": "",
    "focal_spot_enu": [],
    "sun_elevation": 0,
    "sun_azimuth": 0
    }

# Iterate over the batch to perform raytracing and save the results
for n_sample, data in enumerate(batch):
    
    print(f"Running sample {n_sample}/{len(batch)}...")
    
    # Get the required data
    calibration_ids = data['cal_ids']
    sun_elevations = data['sun_elevations']
    sun_azimuths = data['sun_azimuths']
    incident_ray_directions = data['incident_rays']
    target_area_names = data['receiver_targets']
    target_areas = [loaded_scenario.get_target_area(name) for name in target_area_names]
    
    # Set aimpoints to flux center
    loaded_kinematic.aim_points = data['flux_centers']
    
    # Use alignment based on incident rays, to get new motor positions
    heliostat_field.align_surfaces_with_incident_ray_direction(
        incident_ray_direction=data['incident_rays'],
        round_motor_pos=True,
        device=device
        )
    
    # Get orientations and motor positions
    orienations = loaded_kinematic.orientations
    motor_positions = loaded_kinematic.motor_positions
    
    with torch.no_grad():
        bitmaps = align_and_raytrace(scenario=modified_scenario,
                                    incident_ray_directions=incident_ray_directions,
                                    target_areas=target_areas,
                                    align_with_motor_positions=True,
                                    motor_positions=motor_positions,
                                    seed=global_seed,
                                    device=device
                                    )
    
    for h, heliostat_id in enumerate(all_heliostat_ids):
        heliostat_dir = save_dir / f'{heliostat_id}'
        os.makedirs(heliostat_dir, exist_ok=True)
        
        cal_id = int(calibration_ids[h])
        
        bitmap = bitmaps[h]
        target_area = target_areas[h]
        
        center_of_mass = utils.get_center_of_mass(
                                bitmap=bitmap,
                                target_center=target_area.center,
                                plane_e=target_area.plane_e,
                                plane_u=target_area.plane_u,
                                device=device
                            )
        
        surface_normal = orienations[h, 0:4, 2]
        
        properties_dict["surface_normal"] = surface_normal.cpu().tolist()
        properties_dict["motor_position"]["axis_1_motor_position"] = int(motor_positions[h][0].item())
        properties_dict["motor_position"]["axis_2_motor_position"] = int(motor_positions[h][1].item())
        properties_dict["target_name"] = target_area.name
        properties_dict["focal_spot_enu"] = center_of_mass.cpu().tolist()
        properties_dict["sun_elevation"] = sun_elevations[h]
        properties_dict["sun_azimuth"] = sun_azimuths[h]
        
        # Save dictionary to a JSON file
        with open(heliostat_dir/f"{cal_id}-simulation-properties.json", "w") as json_file:
            json.dump(properties_dict, json_file, indent=4)
        
        plt.imshow(bitmap.cpu().detach(), cmap="inferno")
        plt.axis("off")  # Hides both x and y axes
        plt.savefig(heliostat_dir/f"{cal_id}-simulated-flux.png", bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

