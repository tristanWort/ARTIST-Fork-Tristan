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

# Add local path to HeliOptiCal
model_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, model_path) 
from HeliOptiCal.utils.util_simulate import aim_and_shoot_and_save_bitmaps, apply_artificial_blocking, align_and_raytrace, distance_to_blocking_strength
from HeliOptiCal.data_processing.calibration_dataloader import CalibrationDataLoader
from HeliOptiCal.data_processing.calibration_datasplitter import CalibrationDataSplitter
import HeliOptiCal.utils.my_config_dict as my_config_dict
from HeliOptiCal.utils.util import calculate_intersection, normalize_images

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.scenario import Scenario
from artist.util import paint_loader, set_logger_config, utils, config_dictionary
from artist.raytracing import raytracing_utils
import paint.util.paint_mappings as mappings

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
    "/dss/dsshome1/05/di38kid/data/scenarios/20250325_scenario_six_heliostats.h5"
)

# Specify the path for saving the generated data
save_dir = Path("/dss/dsshome1/05/di38kid/data/simulated_data")
name = "6Heliostats"

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

# Perform datasplits
data_splitter = CalibrationDataSplitter(
    metadata_path="/dss/dsshome1/05/di38kid/data/paint/metadata/calibration_metadata_selected_heliostats_20250325_150310.csv",
    output_directory=save_dir / 'splits',
)  
data_splitter.perform_splits(
    training_sizes=[30],
    validation_sizes=[30],
    split_types=["knn"],
    save_splits_plots=False
)
splits = data_splitter.splits
split_df = splits['knn'][(30, 30)]
helio_and_calib_ids = {heliostat_id: split_df.loc[split_df[mappings.HELIOSTAT_ID] == heliostat_id].index.tolist()
                       for heliostat_id in all_heliostat_ids}

shadings = [50, 60, 70, 80, 0, 10, 20, 30, 40]
# Configure dataloader
calibration_data_loader = CalibrationDataLoader(
    data_directory=paint_path,
    heliostats_to_load=all_heliostat_ids,
    power_plant_position=loaded_scenario.power_plant_position,
    load_flux_images=False,
    preload_flux_images=False,
    device=device
)

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

for shading in shadings:
    print("Run with blocking: ", shading/100)
    # Iterate over the batch to perform raytracing and save the results
    for n_sample, data in enumerate(batch):
        
        scenario_dir = save_dir / f"{name}_with_shading_{shading}"
        
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
            bitmaps = normalize_images(bitmaps)
        
        orientations = loaded_kinematic.orientations
        for h, heliostat_id in enumerate(all_heliostat_ids):
            heliostat_dir = scenario_dir / f'{heliostat_id}'
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
            
            # blocked_bitmap, _, _ = (apply_blocking_convolution(
            #     bitmap,
            #     kernel_size=9,
            #     block_strength=0.8,
            #     direction='down',
            #     randomize=True,
            #     random_strength=0.8,
            #     device=device
            #     ))
            
            scale_strength = max(1 - t.item() / 1000, 0.5)
            scale_sigma = min(t.item() / 12.5, 40)
            # print(t.item(), scale_strength, scale_sigma)
            bitmap = apply_artificial_blocking(bitmap, block_ratio=shading / 100, threshold=0.1, sigma=scale_sigma, strength=1)
            
            center_of_mass = utils.get_center_of_mass(
                                bitmap=bitmap,
                                target_center=target_area.center,
                                plane_e=target_area.plane_e,
                                plane_u=target_area.plane_u,
                                device=device
                            )
        
            
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