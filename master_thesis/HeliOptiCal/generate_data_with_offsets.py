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
import sys
import os
import copy
import json
import random

import torch
import torchvision.transforms as transforms
from PIL import Image

from pathlib import Path
from matplotlib import pyplot as plt

from utils_simulate import aim_and_shoot_and_save_bitmaps, align_and_raytrace, apply_blocking_convolution, apply_artificial_blocking

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.scenario import Scenario
from artist.util import paint_loader, set_logger_config, utils, config_dictionary
from artist.raytracing import raytracing_utils
import paint.util.paint_mappings as mappings

# Add local path to HeliOptiCal
model_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, model_path) 
from calibration_dataloader import CalibrationDataLoader
from calibration_datasplitter import CalibrationDataSplitter
import my_config_dict as my_config_dict
from util import calculate_intersection

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
    "/dss/dsshome1/05/di38kid/data/scenarios/250312-1424_scenario_AM35.h5"
)

# Specify the path for saving the generated data
save_dir = Path("/dss/dsshome1/05/di38kid/data/simulated_data/30")

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
modified_scenario = copy.deepcopy(loaded_scenario)

# Get the list of all heliostats in the field
all_heliostat_ids = loaded_scenario.heliostat_field.all_heliostat_names
print('heliostats in field:')
print(all_heliostat_ids)

# Perform datasplits
data_splitter = CalibrationDataSplitter(
    metadata_path="/dss/dsshome1/05/di38kid/data/paint/metadata/calibration_metadata_selected_heliostats_20250325_150310.csv",
    output_directory=save_dir / 'splits',
)  
data_splitter.perform_splits(
    training_sizes=[30],
    validation_sizes=[30],
    split_types=["knn"],
    save_splits_plots=True
)
splits = data_splitter.splits
split_df = splits['knn'][(30, 30)]
helio_and_calib_ids = {heliostat_id: split_df.loc[split_df[mappings.HELIOSTAT_ID] == heliostat_id].index.tolist()
                       for heliostat_id in all_heliostat_ids}

# Configure dataloader
calibration_data_loader = CalibrationDataLoader(
    data_directory=paint_path,
    heliostats_to_load=all_heliostat_ids,
    power_plant_position=loaded_scenario.power_plant_position,
    load_flux_images=False,
    preload_flux_images=False,
    device=device
)

"""Modify the scenario by adding random offsets to its parameters in the kinematic instance."""
# Add random errors to the kinematic parameters.
from add_random_errors import add_random_errors_to_kinematic

error_config = json.load(open('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/simulate_data/error_config.json'))

# Returns a copy of the kinematic instance with changed paramaters
modified_kinematic = add_random_errors_to_kinematic(error_config=error_config,
                                                    scenario=loaded_scenario, 
                                                    save_dir=save_dir / 'parameters', 
                                                    seed=global_seed, 
                                                    device=device)

# Save error config in the folder
json.dump(error_config, open(save_dir / 'parameters/error_config.json', 'w'), indent=4)

# Replace the 'old' kinematic with the modified kinematic
modified_scenario.target_areas = loaded_scenario.target_areas
modified_scenario.heliostat_field.rigid_body_kinematic = modified_kinematic

# For debugging, check if parameter was changed
print(loaded_scenario.heliostat_field.rigid_body_kinematic.all_heliostats_position_params['heliostat_e'][0].data)
print(modified_scenario.heliostat_field.rigid_body_kinematic.all_heliostats_position_params['heliostat_e'][0].data)

# Get the calibration data as a batch   
batch = calibration_data_loader.get_field_batch(helio_and_calib_ids=helio_and_calib_ids, device=device)

properties_dict = {
    config_dictionary.paint_motor_positions: {
        config_dictionary.paint_first_axis: 0,
        config_dictionary.paint_second_axis: 0
        },
    config_dictionary.paint_calibration_target: "",
    my_config_dict.focal_spot_enu_4d: [],
    "original_focal_spot_enu": [],
    my_config_dict.ideal_focal_spot_enu_4d: [],
    config_dictionary.paint_sun_elevation: 0,
    config_dictionary.paint_sun_azimuth: 0
    }

# Iterate over the batch to perform raytracing and save the results
for n_sample, data in enumerate(batch):
    
    print(f"Running sample {n_sample}/{len(batch)}...")
    
    # Get the required data
    calibration_ids = data[my_config_dict.field_sample_calibration_ids]
    sun_elevations = data[my_config_dict.field_sample_sun_elevations]
    sun_azimuths = data[my_config_dict.field_sample_sun_azimuths]
    incident_ray_directions = data[my_config_dict.field_sample_incident_rays]
    target_area_names = data[my_config_dict.field_sample_target_names]
    target_areas = [loaded_scenario.get_target_area(name) for name in target_area_names]
    
    # Set aimpoints to flux center
    loaded_kinematic.aim_points = data[my_config_dict.field_sample_flux_centers]
    
    # Use alignment based on incident rays, to get new motor positions
    heliostat_field.align_surfaces_with_incident_ray_direction(
        incident_ray_direction=incident_ray_directions,
        round_motor_pos=True,
        device=device
        )
    
    # Get orientations and motor positions
    motor_positions = loaded_kinematic.motor_positions
    
    with torch.no_grad():
        bitmaps = align_and_raytrace(scenario=loaded_scenario,
                                    incident_ray_directions=incident_ray_directions,
                                    target_areas=target_areas,
                                    align_with_motor_positions=True,
                                    motor_positions=motor_positions,
                                    seed=global_seed,
                                    device=device
                                    )
    
    orientations = loaded_kinematic.orientations
    for h, heliostat_id in enumerate(all_heliostat_ids):
        heliostat_dir = save_dir / f'{heliostat_id}'
        os.makedirs(heliostat_dir, exist_ok=True)
        
        cal_id = int(calibration_ids[h])
        
        bitmap = bitmaps[h]
        target_area = target_areas[h]
        
        og_center_of_mass = utils.get_center_of_mass(
                                bitmap=bitmap,
                                target_center=target_area.center,
                                plane_e=target_area.plane_e,
                                plane_u=target_area.plane_u,
                                device=device
                            )
        
        surface_normal = orientations[h, 0:4, 2]
        reflected_ray = raytracing_utils.reflect(
                        incident_ray_directions[h], 
                        surface_normal
                        )
        ideal_center_point, t = calculate_intersection(
                        ray_origin=orientations[h, 0:4, 3],
                        ray_direction=reflected_ray,
                        plane_center=target_area.center,
                        plane_normal=target_area.normal_vector,
                    )
        
        random_number = random.randint(1, 3)
        if random_number == 3:
            print(f"Blocked: {cal_id}")
            blocked_bitmap, _, _ = (apply_blocking_convolution(
                bitmap,
                kernel_size=9,
                block_strength=0.8,
                direction='down',
                randomize=True,
                random_strength=0.8,
                device=device
                ))
            center_of_mass = utils.get_center_of_mass(
                                bitmap=blocked_bitmap,
                                target_center=target_area.center,
                                plane_e=target_area.plane_e,
                                plane_u=target_area.plane_u,
                                device=device
                            )
            bitmap = blocked_bitmap
            
        else:
            center_of_mass = og_center_of_mass
        
        properties_dict["original_focal_spot_enu"] = og_center_of_mass.cpu().tolist()
        properties_dict[my_config_dict.ideal_focal_spot_enu_4d] = ideal_center_point.cpu().tolist()
        properties_dict[config_dictionary.paint_motor_positions][config_dictionary.paint_first_axis] = int(motor_positions[h][0].item())
        properties_dict[config_dictionary.paint_motor_positions][config_dictionary.paint_second_axis] = int(motor_positions[h][1].item())
        properties_dict[config_dictionary.paint_calibration_target] = target_area.name
        properties_dict[my_config_dict.focal_spot_enu_4d] = center_of_mass.cpu().tolist()
        properties_dict[config_dictionary.paint_sun_elevation] = sun_elevations[h].item()
        properties_dict[config_dictionary.paint_sun_azimuth] = sun_azimuths[h].item()
        
        # Save dictionary to a JSON file
        os.makedirs(heliostat_dir/f"Calibration", exist_ok=True) 
        with open(heliostat_dir/f"Calibration/{cal_id}-calibration-properties.json", "w") as json_file:
            json.dump(properties_dict, json_file, indent=4)
        
        
        width_in_pixels, height_in_pixels = 512, 512
        dpi = 100  # Can be any value
        width_in_inches = width_in_pixels / dpi
        height_in_inches = height_in_pixels / dpi

        plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.imshow(bitmap.cpu().detach(), cmap="gray")
        plt.axis("off")  # Hides both x and y axes
        plt.savefig(heliostat_dir/f"Calibration/{cal_id}-flux.png", bbox_inches="tight", pad_inches=0, dpi=dpi)
        plt.close()