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
from HeliOptiCal.utils.util_simulate import (aim_and_shoot_and_save_bitmaps, apply_artificial_blocking)
from HeliOptiCal.data_processing.calibration_dataloader import CalibrationDataLoader
from HeliOptiCal.data_processing.calibration_datasplitter import CalibrationDataSplitter
import HeliOptiCal.utils.my_config_dict as my_config_dict
from HeliOptiCal.utils.util import calculate_intersection, get_bitmap_indices_from_center_coordinates

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


def align_and_raytrace(scenario,
                       incident_ray_directions: torch.tensor,
                       target_areas: list(),
                       aim_points=None,
                       align_with_motor_positions=False,
                       motor_positions=None,
                       seed=42,
                       device='cuda'):
        """
        Perform aiming and raytracing for the heliostat field for the given scenario.
        
        For alignment there are two options.
            1) Align with incident ray directions and aim points.
            2) Align with motor positions.
        """
        # Auxiliary referencing to kinematic
        kinematic = scenario.heliostat_field.rigid_body_kinematic
        
        if not align_with_motor_positions:
            # Set aim points and then align with incident ray directions
            kinematic.aim_points = aim_points
            # Align with incident ray directions
            scenario.heliostat_field.align_surfaces_with_incident_ray_direction(
                incident_ray_direction=incident_ray_directions,
                round_motor_pos=True,
                device=device
                )
        
        else:
            # Align with motor positions
            scenario.heliostat_field.align_surfaces_with_motor_positions(
               motor_positions=motor_positions,
               device=device
               )
        
        # Initiate Raytracer
        raytracer = HeliostatRayTracer(scenario=scenario, 
                                       world_size=1, 
                                       rank=0, 
                                       batch_size=1,
                                       bitmap_resolution_e=960,
                                       bitmap_resolution_u=960,
                                       random_seed=seed) 
        # Raytrace and store bitmaps
        bitmaps = raytracer.trace_rays_separate(incident_ray_directions=incident_ray_directions,
                                                      target_areas=target_areas,
                                                      device=device)
        # bitmaps = raytracer.normalize_bitmaps(bitmaps, target_areas)
        
        return bitmaps


def plot_center_cross(center_indices: (int, int), cross_size: int = 20, cross_color: str = "red", cross_linewidth: float = 1.5, label: str = None):
    
    u_idx, e_idx = center_indices
    # Draw horizontal and vertical lines to form a cross
    plt.plot(
        [e_idx - cross_size, e_idx + cross_size],
        [u_idx, u_idx],
        color=cross_color,
        linewidth=cross_linewidth,
        label=label if label else "_nolegend_"  # Avoid duplicate legends
    )
    plt.plot(
        [e_idx, e_idx],
        [u_idx - cross_size, u_idx + cross_size],
        color=cross_color,
        linewidth=cross_linewidth,
    )
    

def save_bitmap_with_center_crosses(bitmap: torch.Tensor, center_indices: (int, int), save_under: Path, tight_layout=True, cross_legend=True, title=None):
    """
    Save a bitmap image with a red cross marking the predicted center.

    Parameters
    ----------
    bitmap : torch.Tensor
        The bitmap image to be saved (2D tensor).
    center_indices : Tuple[int, int]
        The (u_idx, e_idx) center of the flux bitmap to mark with a cross.
    save_under : Path
        The path to save the image file.
    """
    width_in_pixels, height_in_pixels = 960, 960
    dpi = 100
    width_in_inches = width_in_pixels / dpi
    height_in_inches = height_in_pixels / dpi

    plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
    plt.imshow(bitmap.cpu().detach(), cmap="gray")
    
    # plot_center_cross(center_indices[0], cross_color='red')
    
    plt.plot(
    center_indices[0][1], center_indices[0][0],
    color='red', marker='x', markersize=30,
    )
    plot_center_cross(center_indices[1], cross_color='blue', cross_size=20)
    # plt.plot(
    #     center_indices[1][1], center_indices[1][0],
    #     color='blue', marker='x', markersize=20,
    # )
    plt.plot([], [], color='red', marker='x', linestyle='None', markersize=14, label='Unit Prediction of Centroid')
    plt.plot([], [], color='blue', marker='x', linestyle='None', markersize=14, label='Ideal Centroid on Reflection Axis')

    if title:
        plt.title(title, fontsize=18, pad=10)
        pad_inches=0.12
    else:
        pad_inches=0
    if cross_legend:
        plt.legend(loc='upper right', fontsize=14)
    plt.axis("off")
    
    if tight_layout:
        plt.savefig(save_under, bbox_inches="tight", pad_inches=pad_inches, dpi=dpi)
    else:
        plt.savefig(save_under, dpi=dpi)
    plt.close()
    

# Set up logger
set_logger_config()

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(
    "/dss/dsshome1/05/di38kid/data/scenarios/250129-1347_scenario_AA39.h5"
)

# Specify the path for saving the generated data
save_dir = Path("/dss/dsshome1/05/di38kid/data/simulated_data/SPIE_PAPER/surface_imperfections")
os.makedirs(save_dir, exist_ok=True)

# Specify path to paint-data
paint_path = pathlib.Path("/dss/dsshome1/05/di38kid/data/paint")

# Load two scenarios.
with h5py.File(scenario_path) as scenario_file:
    scenario_AA39 = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

scenario_path = pathlib.Path(
    "/dss/dsshome1/05/di38kid/data/scenarios/20250325_scenario_six_heliostats.h5"
)
with h5py.File(scenario_path) as scenario_file:
    scenario_six_heliostats = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file, device=device
    )

six_heliostats_field =  scenario_six_heliostats.heliostat_field
six_positions = scenario_six_heliostats.heliostat_field.all_heliostat_positions
six_kinematic = six_heliostats_field.rigid_body_kinematic

AA39_heliostat_field = scenario_AA39.heliostat_field 
AA39_kinematic = AA39_heliostat_field.rigid_body_kinematic

# Get the list of all heliostats in the field
all_heliostat_ids = scenario_six_heliostats.heliostat_field.all_heliostat_names
print('heliostats in field:')
print(all_heliostat_ids)

# for h_idx, heliostat in enumerate(all_heliostat_ids):
#     six_heliostats_field.all_surface_normals[h_idx] = AA39_heliostat_field.all_surface_normals[0]
#     six_heliostats_field.all_surface_points[h_idx] = AA39_heliostat_field.all_surface_points[0]

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

# Configure dataloader
calibration_data_loader = CalibrationDataLoader(
    data_directory=paint_path,
    heliostats_to_load=all_heliostat_ids,
    power_plant_position=scenario_AA39.power_plant_position,
    load_flux_images=False,
    preload_flux_images=False,
    device=device
)

# Get the calibration data as a batch   
batch = calibration_data_loader.get_field_batch(helio_and_calib_ids=helio_and_calib_ids, device=device)
                         
for n_sample, data in enumerate(batch[:1]):
        
    print(f"Running sample {n_sample}/{len(batch)}...")
    
    # Get the required data
    calibration_ids = data[my_config_dict.field_sample_calibration_ids]
    sun_elevations = data[my_config_dict.field_sample_sun_elevations]
    sun_azimuths = data[my_config_dict.field_sample_sun_azimuths]
    incident_ray_directions = data[my_config_dict.field_sample_incident_rays]
    target_area_names = data[my_config_dict.field_sample_target_names]
    target_area = scenario_six_heliostats.get_target_area(target_area_names[0])
    
    aim_points = []
    target_areas = []
    for heliostat in all_heliostat_ids:
        target_areas.append(target_area)
        aim_points.append(target_area.center)
    aim_points = torch.stack(aim_points)

    # Set aimpoints to flux center
    six_kinematic.aim_points = aim_points
    
    # Use alignment based on incident rays, to get new motor positions
    six_heliostats_field.align_surfaces_with_incident_ray_direction(
        incident_ray_direction=incident_ray_directions,
        round_motor_pos=False,
        device=device
        )
    
    with torch.no_grad():
        bitmaps = align_and_raytrace(scenario=scenario_six_heliostats,
                                    incident_ray_directions=incident_ray_directions,
                                    target_areas=target_areas,
                                    aim_points=aim_points,
                                    align_with_motor_positions=False,
                                    seed=global_seed,
                                    device=device
                                    )
    
    orientations = six_kinematic.orientations
    for h, heliostat_id in enumerate(all_heliostat_ids):
        
        cal_id = int(calibration_ids[h])
        
        bitmap = bitmaps[h]
        
        og_center_of_mass = utils.get_center_of_mass(
                                bitmap=bitmap,
                                target_center=target_area.center,
                                plane_e=target_area.plane_e,
                                plane_u=target_area.plane_u,
                                device=device
                            )
        
        og_indices = get_bitmap_indices_from_center_coordinates(bitmap, og_center_of_mass, target_area.center, target_area.plane_e, target_area.plane_u, device)
        
        surface_normal = orientations[h, 0:4, 2]
        reflected_ray = raytracing_utils.reflect(incident_ray_directions[h], surface_normal)
        ideal_center_point, t = calculate_intersection(orientations[h, 0:4, 3], reflected_ray, target_area.center, target_area.normal_vector)
        
        print("Distance from Heliostat to Target =", t.item(), "m")        
        ideal_indices = get_bitmap_indices_from_center_coordinates(bitmap, ideal_center_point, target_area.center, target_area.plane_e, target_area.plane_u, device)
        save_bitmap_with_center_crosses(bitmap, [og_indices, ideal_indices], save_dir / f'{heliostat_id}_error_of_center_of_mass__title_and_legend.png', 
                                        tight_layout=True, cross_legend=True, title=f'Heliostat Distance from Target: {t.item() :.2f} m')
        save_bitmap_with_center_crosses(bitmap, [og_indices, ideal_indices], save_dir / f'{heliostat_id}_error_of_center_of_mass__legend.png', 
                                        cross_legend=True, tight_layout=True)
        save_bitmap_with_center_crosses(bitmap, [og_indices, ideal_indices], save_dir / f'{heliostat_id}_error_of_center_of_mass__raw.png', 
                                        cross_legend=False, tight_layout=True)
