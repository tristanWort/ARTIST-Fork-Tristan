"""
This script serves for generating plots which visualize the distribution of errors in COM-reflection axis vs GT reflection axis.
The plot will show all samples' angular errors in mrad in a given dataset for a B&S simulated scenario.  
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
from typing import List

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
from HeliOptiCal.plot_results.plot_errors_distributions import (load_sun_positions, merge_data, 
                                                                plot_alignment_errors_over_sun_pos, 
                                                                plot_error_violinplots_by_scenario, 
                                                                plot_errors_over_heliostats_by_distance,
                                                                plot_error_kde_bands_over_heliostats_by_distance
                                                                )

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
device = torch.device("cpu")

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    "/dss/dsshome1/05/di38kid/data/scenarios/20250525_scenario_20_heliostats.h5"
)

def analyze_reflection_axis_errors_in_datasets(dataset_dir: str):
    # Specify the path for saving the generated data
    name = "20Heliostats"

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

    # Load sun positions
    metadata_file ="/dss/dsshome1/05/di38kid/data/paint/selected_20/metadata/calibration_metadata_selected_heliostats_20250525_161028.csv"
    sun_positions_df = load_sun_positions(metadata_file)

    # Perform datasplits
    data_splitter = CalibrationDataSplitter(
        metadata_path=metadata_file,
        output_directory=Path(dataset_dir) / 'splits',
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
    calibration_data_loader = CalibrationDataLoader(data_directory=dataset_dir,
                                                    heliostats_to_load=all_heliostat_ids,
                                                    power_plant_position=loaded_scenario.power_plant_position,
                                                    is_simulated_data=True,
                                                    load_flux_images=False,
                                                    preload_flux_images=False,
                                                    device=device)

    # Get the calibration data as a batch   
    # batch = calibration_data_loader.get_field_batch(helio_and_calib_ids=helio_and_calib_ids, device=device)

    modes = ['train', 'validation', 'test']
    error_data = []
    heliostat_distances = {hel_id: 0 for hel_id in all_heliostat_ids}

    for mode in modes:
        print(f"Run trought mode {mode}")
        split_ids = data_splitter.get_helio_and_calib_ids_from_split('knn', (30, 30), mode, heliostat_ids=all_heliostat_ids)
        batch = calibration_data_loader.get_field_batch(helio_and_calib_ids=split_ids, device=device)
        # Iterate over the batch to perform raytracing and save the results
        for n_sample, data in enumerate(batch):
            print(f"\tRunning sample {n_sample}/{len(batch)}...")
            
            # Get the required data
            calibration_ids = data[my_config_dict.field_sample_calibration_ids]
            sun_elevations = data[my_config_dict.field_sample_sun_elevations]
            sun_azimuths = data[my_config_dict.field_sample_sun_azimuths]
            incident_ray_directions = data[my_config_dict.field_sample_incident_rays]
            target_area_names = data[my_config_dict.field_sample_target_names]
            target_areas = [loaded_scenario.get_target_area(name) for name in target_area_names]
            motor_positions=data[my_config_dict.field_sample_motor_positions]
            
            # Get flux centers and ideal flux centers
            flux_centers = data[my_config_dict.field_sample_flux_centers]
            ideal_flux_centers = data[my_config_dict.field_sample_ideal_flux_centers]
            
            # Use alignment based on incident rays, to get new motor positions
            heliostat_field.align_surfaces_with_motor_positions(
                motor_positions=motor_positions,
                device=device
                )
            orientations = loaded_kinematic.orientations
            
            # Iterate over heliostats' samples and calulate error in reflection axis
            for h, heliostat_id in enumerate(all_heliostat_ids):
                    
                cal_id = int(calibration_ids[h])
                # target_area = target_areas[h]
                
                # compute error in approximated reflection axis vs GT reflection axis
                surface_midpoint = orientations[h, 0:4, 3]
                if n_sample == 0:
                    heliostat_distances[heliostat_id] = torch.sqrt(surface_midpoint[0] **2 
                                                                   + surface_midpoint[1] **2).item()
                
                vect1 = (flux_centers[h] - surface_midpoint)
                vect2 = (ideal_flux_centers[h] - surface_midpoint)
                m1 = torch.norm(vect1, dim=-1, dtype=torch.float64)
                m2 = torch.norm(vect2, dim=-1, dtype=torch.float64)
                dot_product = torch.sum(vect1 * vect2, dim=-1)
                cos_sim = (dot_product + 1e-12) / (m1 * m2 + 1e-12)
                angle_rad = torch.acos(torch.clamp(cos_sim, min= -1.0, max= 1.0)).abs()
                error_in_mrad = angle_rad * 1000

                mode_mapping = {'train': 'Training', 'validation': 'Validation', 'test': 'Testing'}
                error_data.append({
                        'heliostat_id': heliostat_id,
                        'mode': mode_mapping[mode],
                        'calib_id': cal_id,
                        'error': error_in_mrad.detach().numpy() 
                    })
        
    error_df = pd.DataFrame(error_data)
    merged_data = merge_data(error_df, sun_positions_df)
    return merged_data, heliostat_distances


def load_reflection_axis_errors_for_mult_datasets(datastet_dirs: List[str]):
    merged_data = []
    for dataset_dir in datastet_dirs:
        data, _ = analyze_reflection_axis_errors_in_datasets(dataset_dir)
        merged_data.append(data)
    return merged_data
    
"""Compute errors in reflection axis for single dataset and generate sun positions plots."""
dataset = Path("/dss/dsshome1/05/di38kid/data/simulated_data/20Heliostats_04/20Heliostats_with_realistic_bs")
# reflection_error_df, heliostat_distances = analyze_reflection_axis_errors_in_datasets(dataset)
save_dir = Path("/dss/dsshome1/05/di38kid/data/results/plots/methodology/realistic_bs")
# plot_alignment_errors_over_sun_pos(reflection_error_df, output_dir=save_dir, error_label='Error in GT-Reflection Axis', sep_plots=False, print_mean_error=False, marker_for_training='o')
# plot_errors_over_heliostats_by_distance(reflection_error_df, heliostat_distances, save_dir/'realistic_distances_over_elevation.pdf', 
#                                         marker_for_training='o', print_mean_error=False)
# plot_error_kde_bands_over_heliostats_by_distance(reflection_error_df, heliostat_distances, save_dir/'base_bands_distances_over_elevation.pdf', 
#                                                  marker_for_training='o', print_mean_error=False)


"""Collect errors in reflection axis for multiple datasets and generate violin plot."""
dataset_dirs = [
    "/dss/dsshome1/05/di38kid/data/simulated_data/20Heliostats_04/20Heliostats_base",
    "/dss/dsshome1/05/di38kid/data/simulated_data/20Heliostats_04/20Heliostats_with_realistic_bs",
    "/dss/dsshome1/05/di38kid/data/simulated_data/20Heliostats_01/20Heliostats_with_blocking_20_on_4th",
    "/dss/dsshome1/05/di38kid/data/simulated_data/20Heliostats_01/20Heliostats_with_blocking_20_on_3rd",
    "/dss/dsshome1/05/di38kid/data/simulated_data/20Heliostats_01/20Heliostats_with_blocking_40_on_4th",
    "/dss/dsshome1/05/di38kid/data/simulated_data/20Heliostats_01/20Heliostats_with_blocking_40_on_3rd",
    "/dss/dsshome1/05/di38kid/data/simulated_data/20Heliostats_01/20Heliostats_with_blocking_60_on_4th",
    "/dss/dsshome1/05/di38kid/data/simulated_data/20Heliostats_01/20Heliostats_with_blocking_60_on_3rd"
]
reflection_error_dfs = load_reflection_axis_errors_for_mult_datasets(dataset_dirs)
scenario_names = ["Base", "Realistic", "20%_on_4th", "20%_on_3rd", "40%_on_4th", "40%_on_3rd", "60%_on_4th", "60%_on_3rd",]
save_dir = Path("/dss/dsshome1/05/di38kid/data/results/plots/methodology/blocking")
plot_error_violinplots_by_scenario(
    scenario_dfs=reflection_error_dfs,
    scenario_labels=scenario_names,
    output_dir=save_dir,
    y_label="COM-Vector vs. Reflection Axis [mrad]",
    plot_title=None,
)

