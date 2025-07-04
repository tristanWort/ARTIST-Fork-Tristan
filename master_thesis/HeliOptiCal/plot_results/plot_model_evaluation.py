"""
Script for generation example plots for methodology of model evaluation.
The model output is evaluated based on the error in alignment and on contours (prediction vs ground truth).
This script generates 
(1) One diff-plot indicating the error in alignment, by showing position of flux centers.
(2) One diff-plot showing the relative positions of the predicted and true contour.
"""

import torch
import os
import sys
import json
import h5py
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 
from artist.util import config_dictionary, paint_loader
from artist.util.utils import get_center_of_mass
from artist.util.scenario import Scenario
from artist.scene.light_source_array import LightSourceArray
from artist.scene.sun import Sun

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, repo_path)
from HeliOptiCal.data_generation.generate_scenario import build_heliostat_file_list
from HeliOptiCal.utils.util_simulate import align_and_raytrace, save_bitmap_with_2_center_crosses, gaussian_filter_2d, save_bitmap, save_bitmap_with_center_cross
from HeliOptiCal.utils.util import normalize_images, normalize_and_interpolate, get_bitmap_indices_from_center_coordinates, calculate_intersection
from HeliOptiCal.image_losses.image_loss import find_soft_contour_pytorch_vertical


# Set the device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the scenario.
scenario_file = "/dss/dsshome1/05/di38kid/data/scenarios/250129-1347_scenario_AA39.h5"
with h5py.File(scenario_file, "r") as f:
    example_scenario = Scenario.load_scenario_from_hdf5(scenario_file=f, device=device)
# load power plant position
power_plant_position = example_scenario.power_plant_position
# load heliostat ids
all_heliostat_ids = example_scenario.heliostat_field.all_heliostat_names


def change_sun_distribution(scenario, sun_mean=0.0, sun_cov=2*4.3681e-6, device="cuda"):
    """Change the sun distribution parameters in the ARTIST scenario."""
    mean = torch.tensor([sun_mean, sun_mean], dtype=torch.float64, device=device)
    cov = torch.tensor([[sun_cov, 0], [0, sun_cov]], dtype=torch.float64, device=device)
    distribution = torch.distributions.MultivariateNormal(mean, cov)
    sun = scenario.light_sources.light_source_list[0]
    sun.distribution = distribution
    return scenario


# Change sun distribution in the scenario
# example_scenario = change_sun_distribution(example_scenario, sun_cov=2*4.3681e-6, device=device)

# Load calibration data
heliostat_id = all_heliostat_ids[0]
calibration_id = "163373"
calibration_properties_path = Path("/dss/dsshome1/05/di38kid/data/paint/selected_20/AA39/Calibration/163373-calibration-properties.json")
target_name, _ , sun_position, motor_positions = paint_loader.extract_paint_calibration_data(
        calibration_properties_paths=[calibration_properties_path],
        power_plant_position=example_scenario.power_plant_position,
        device=device
        )
target_area = example_scenario.get_target_area(target_name[0])
target_areas = [target_area]
incident_ray_directions = (torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_position)
motor_positions += torch.tensor([200, 500], device=device)

# Load ground truth bitmap
flux_path = Path("/dss/dsshome1/05/di38kid/data/paint/selected_20/AA39/Calibration/163373-flux.png")
flux_image = Image.open(flux_path)
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((256, 256)),
                                transforms.ToTensor()])
true_bitmaps = transform(flux_image).to(device)
norm_true_bitmaps = normalize_images(true_bitmaps)
true_center_coords = get_center_of_mass(norm_true_bitmaps[0], target_area.center, target_area.plane_e, target_area.plane_u, device=device)
true_center_indices = get_bitmap_indices_from_center_coordinates(norm_true_bitmaps[0], true_center_coords, target_area.center, target_area.plane_e, target_area.plane_u, device)

# Align and raytrace
pred_bitmaps = align_and_raytrace(example_scenario, incident_ray_directions, target_areas, align_with_motor_positions=True, motor_positions=motor_positions, device=device)
norm_pred_bitmaps = normalize_images(pred_bitmaps)
pred_center_coords = get_center_of_mass(norm_pred_bitmaps[0], target_area.center, target_area.plane_e, target_area.plane_u, device=device)
pred_center_indices = get_bitmap_indices_from_center_coordinates(norm_pred_bitmaps[0], pred_center_coords, target_area.center, target_area.plane_e, target_area.plane_u, device)

# Generate output plots
save_here = Path("/dss/dsshome1/05/di38kid/data/results/plots/methodology")
save_bitmap(norm_true_bitmaps[0], save_here / f'{heliostat_id}_{calibration_id}_True_Flux.png')
save_bitmap(norm_pred_bitmaps[0], save_here / f'{heliostat_id}_{calibration_id}_Pred_Flux.png')
save_bitmap_with_center_cross(norm_true_bitmaps[0], true_center_indices, save_here / f'{heliostat_id}_{calibration_id}_True_Flux_with_Center.png')
save_bitmap_with_center_cross(norm_pred_bitmaps[0], pred_center_indices, save_here / f'{heliostat_id}_{calibration_id}_Pred_Flux_with_Center.png')
# Save a diff plot for flux with center crosses
save_bitmap_with_2_center_crosses(norm_pred_bitmaps[0]-norm_true_bitmaps[0], [pred_center_indices, true_center_indices], save_here / f'{heliostat_id}_{calibration_id}_Diff_Flux_with_Centers.png')


def apply_gaussian_find_contours(input_bitmaps, sharpness=80, threshold=0.3, sigma_in=10, sigma_out=15, num_interpolations=4):
    """
    Function for applying gaussian blurring to input bitmap and returning the upper contour.
    """
    # First normalize inputs
    input_bitmaps = normalize_and_interpolate(input_bitmaps, num_interpolations).squeeze(0)
    gauss_bitmaps = torch.empty_like(input_bitmaps)
    gauss_contours = torch.empty_like(input_bitmaps)
    for i in range(input_bitmaps.shape[0]):
        # Apply Gaussian filter on inputs, save and return
        gauss_img = gaussian_filter_2d(input_bitmaps[i], sigma=sigma_in)
        gauss_bitmaps[i] = normalize_images(gauss_img.unsqueeze(0))
        # Find contours, apply Gaussian filter on output
        contour = find_soft_contour_pytorch_vertical(gauss_bitmaps[i], threshold=threshold, sharpness=sharpness)
        gauss_contours[i] = gaussian_filter_2d(contour.squeeze(0), sigma=sigma_out)
    # Normalize output
    contours = normalize_and_interpolate(gauss_contours, num_interpolations).squeeze(0)
    return contours, input_bitmaps

# Extract contours for groundtruth and predicted flux bitmaps
true_contour, _ = apply_gaussian_find_contours(norm_true_bitmaps)
pred_contour, _ = apply_gaussian_find_contours(norm_pred_bitmaps)
save_bitmap(true_contour[0], save_here / f'{heliostat_id}_{calibration_id}_True_Contour.png')
save_bitmap(pred_contour[0], save_here / f'{heliostat_id}_{calibration_id}_Pred_Contour.png')
save_bitmap(pred_contour[0]-true_contour[0], save_here / f'{heliostat_id}_{calibration_id}_Diff_Contour.png')

