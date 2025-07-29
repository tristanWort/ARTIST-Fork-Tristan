"""
This is a script for generating data which can used to validate alignment / calibration.

The data which is generated consists of json-files and flux-images.
Based on real sun and motor positions which are extracted from PAINT-data, the simulated
datasets are generated using the ARTIST's Raytracer. 
"""
import pathlib
import time
import h5py
import sys
import os
import copy
import json
import random

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

from pathlib import Path
from matplotlib import pyplot as plt

# Add local path to HeliOptiCal
model_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, model_path) 
from HeliOptiCal.utils.util_simulate import (aim_and_shoot_and_save_bitmaps,
                                             estimate_blocking_factor,
                                             estimate_shading_factor, 
                                             compute_sigma_pixels,
                                             apply_artificial_blocking, 
                                             align_and_raytrace)
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
    "/dss/dsshome1/05/di38kid/data/scenarios/20250525_scenario_20_heliostats.h5"
)

# Specify the path for saving the generated data
name = "20Heliostats"
save_dir = Path("/dss/dsshome1/05/di38kid/data/simulated_data/20Heliostats_04")

# Specify path to paint-data
paint_path = pathlib.Path("/dss/dsshome1/05/di38kid/data/paint/selected_20")

# Load two scenarios.
with h5py.File(scenario_path) as scenario_file:
    loaded_scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )
heliostat_field = loaded_scenario.heliostat_field 
loaded_kinematic = heliostat_field.rigid_body_kinematic

# Get the list of all heliostats in the field
all_heliostat_ids = loaded_scenario.heliostat_field.all_heliostat_names
print('heliostats in field:')
print(all_heliostat_ids)

# Perform datasplits
data_splitter = CalibrationDataSplitter(
    metadata_path=paint_path / "metadata/calibration_metadata_selected_heliostats_20250525_161028.csv",
    output_directory=paint_path / 'splits',
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

# Configure dataloader
calibration_data_loader = CalibrationDataLoader(data_directory=paint_path,
                                                heliostats_to_load=all_heliostat_ids,
                                                power_plant_position=loaded_scenario.power_plant_position,
                                                load_flux_images=False,
                                                preload_flux_images=False,
                                                device=device)

# Get the calibration data as a batch   
batch = calibration_data_loader.get_field_batch(helio_and_calib_ids=helio_and_calib_ids, device=device)

# Setup of an empty shell for storing calibration measurement data
properties_dict = {
    config_dictionary.paint_motor_positions: {
        config_dictionary.paint_first_axis: 0,
        config_dictionary.paint_second_axis: 0
        },
    config_dictionary.paint_calibration_target: "",
    my_config_dict.focal_spot_enu_4d: [],
    my_config_dict.unblocked_focal_spot_enu_4d: [],
    my_config_dict.ideal_focal_spot_enu_4d: [],
    config_dictionary.paint_sun_elevation: 0,
    config_dictionary.paint_sun_azimuth: 0,
    }

"""Load calibration metadata to get all heliostat positions."""
heliostats_properties_metadata = '/dss/dsshome1/05/di38kid/data/paint/metadata/properties_metadata_all_heliostats.csv'
df_properties = pd.read_csv(heliostats_properties_metadata)
df_properties = df_properties[['HeliostatId', 'latitude', 'longitude']].dropna()
df_properties = df_properties.drop_duplicates(subset='HeliostatId', keep='first').reset_index(drop=True)


def to_enu(df, receiver_latlon):
    enu_list = []
    for _, row in df.iterrows():
        coords = torch.tensor([row["latitude"], row["longitude"], 0.0], dtype=torch.float64)
        enu = utils.convert_wgs84_coordinates_to_local_enu(coords, receiver_latlon, device)
        enu_list.append(enu[:2].tolist())  # east, north
    df[["east", "north"]] = pd.DataFrame(enu_list, index=df.index)
    return df


def get_north_measurements(heliostat_id: str) -> (float, float):
    """
    Given a heliostat ID, finds the heliostat in front of it (one row ahead alphabetically),
    and returns the spacing in the north direction between them, as well as the rear heliostat's 
    north coordinate.

    Parameters
    ----------
    heliostat_id : str
        ID of the heliostat (e.g., 'BB29')

    Returns
    -------
    float
        Distance in the north direction between the heliostat and its front neighbor.
        Returns None if the front heliostat is not found.
    """
    assert isinstance(heliostat_id, str) and len(heliostat_id) >= 4, "Invalid Heliostat ID format"
    
    row_letters = heliostat_id[:2]
    col_index = heliostat_id[2:]
            
    # Early exit for the first row 'AA'
    if row_letters == 'AA':
        return None
    
    # Step 1: compute the row in front
    row_front = list(row_letters)
    if row_front[1] == 'A':
        row_front[0] = chr(ord(row_front[0]) - 1)
        row_front[1] = 'Z'
    else:
        row_front[1] = chr(ord(row_front[1]) - 1)
    front_id = "".join(row_front) + col_index

    # Step 2: extract positions
    current_row = df_properties[df_properties["HeliostatId"] == heliostat_id]
    if current_row.empty:
        print(f"Warning: Could not find '{heliostat_id}' in the data.")
        return None
    front_row = df_properties[df_properties["HeliostatId"] == front_id]
    if front_row.empty:
        print(f"Warning: Could not find '{front_id}' in the data.")
        return None

    north_current = float(current_row.iloc[0]["north"])
    north_front = float(front_row.iloc[0]["north"])

    spacing = torch.tensor(abs(north_current - north_front), dtype=torch.float32)
    return spacing


# Convert coordinates to EN
power_plant_position = loaded_scenario.power_plant_position
df_properties = to_enu(df_properties, power_plant_position)

# Height of mirror surface is constant across all heliostats
h0 = torch.tensor(3.22, dtype=torch.float32)
        
scenario_dir = save_dir / f"{name}_with_realistic_bs"

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
                                    resolution_eu=(960, 960),
                                    seed=global_seed,
                                    device=device)
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
                                threshold=0.1,
                                device=device)
        
        surface_normal = orientations[h, 0:4, 2]
        reflected_ray = raytracing_utils.reflect(
                        incident_ray_directions[h], 
                        surface_normal)
        
        ideal_center_point, t = calculate_intersection(
                        ray_origin=orientations[h, 0:4, 3],
                        ray_direction=reflected_ray,
                        plane_center=target_area.center,
                        plane_normal=target_area.normal_vector)
        
        # compute geometric input parameters for blocking and shading
        translation_to_target = target_area.center - orientations[h, 0:4, 3]
        spacing = get_north_measurements(heliostat_id)
        
        # estimate factors for relative surface height which is blocked & shaded
        if spacing is None:
            max_bs = None
            f_bs = 0.0
        else:
            f_b = estimate_blocking_factor(sun_elevations[h], h0, spacing, translation_to_target)
            f_s = estimate_shading_factor(sun_elevations[h], h0, spacing, translation_to_target)
            max_bs = max(f_b / 2, f_s)
            # f_bs = max_bs
            f_bs = random.uniform(0.0, max_bs)
        
        sigma = compute_sigma_pixels(t.item(), target_area.plane_u, 960)
        if f_bs != 0:
            print(f"\t[{heliostat_id}] [{sun_elevations[h]}][ {t.item()}] with {f_bs} [max {max_bs}] and sigma {sigma}.")
            bitmap = apply_artificial_blocking(bitmap, block_ratio=f_bs, threshold=0.15, sigma=sigma, strength=1)
        
        else:
            print(f"\t[{heliostat_id}] [{sun_elevations[h]}][ {t.item()}] no B/S.") 
            
        center_of_mass = utils.get_center_of_mass(bitmap=bitmap,
                                                target_center=target_area.center,
                                                plane_e=target_area.plane_e,
                                                plane_u=target_area.plane_u,
                                                threshold=0.1,
                                                device=device)
    
        
        properties_dict[my_config_dict.unblocked_focal_spot_enu_4d] = og_center_of_mass.cpu().tolist()
        properties_dict[my_config_dict.ideal_focal_spot_enu_4d] = ideal_center_point.cpu().tolist()
        properties_dict[config_dictionary.paint_motor_positions][config_dictionary.paint_first_axis] = int(motor_positions[h][0].item())
        properties_dict[config_dictionary.paint_motor_positions][config_dictionary.paint_second_axis] = int(motor_positions[h][1].item())
        properties_dict[config_dictionary.paint_calibration_target] = target_area.name
        properties_dict[my_config_dict.focal_spot_enu_4d] = center_of_mass.cpu().tolist()
        properties_dict[config_dictionary.paint_sun_elevation] = sun_elevations[h].item()
        properties_dict[config_dictionary.paint_sun_azimuth] = sun_azimuths[h].item()
        properties_dict["f_bs"] = f_bs
        properties_dict["sigma_bs"] = sigma
        
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
        