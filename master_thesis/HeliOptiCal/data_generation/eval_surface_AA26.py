"""
Within this script the distortions in the COM-vector from the reflection axis will be evluated for heliostat AA26.

Three scenarios will be evaluated by raytracing a comparing the COM coordinate to the intersect point of the reflection axis:
(1) Load planar ideal surface & align east-coordinate of heliostat with target tower center.
(2) Load deflectometry data as surface & align east-coordinate of heliostat with target tower center.
(3) Finally, set actual heliostat position. 
"""
import pathlib
import os
import sys
import time
import torch
import h5py

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 
from artist.util import config_dictionary, paint_loader, set_logger_config, utils
from artist.raytracing import raytracing_utils
from artist.util.configuration_classes import (
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.util.scenario_generator import ScenarioGenerator
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.scenario import Scenario

# Add local path to HeliOptiCal
model_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, model_path) 
from HeliOptiCal.data_generation.plots_surface_imperfections import align_and_raytrace, plot_center_cross, save_bitmap_with_center_crosses
from HeliOptiCal.utils.util import calculate_intersection, get_bitmap_indices_from_center_coordinates, normalize_images
from HeliOptiCal.utils.util_simulate import (estimate_blocking_factor,
                                             estimate_shading_factor, 
                                             compute_sigma_pixels,
                                             apply_artificial_blocking)


# Set up logger
set_logger_config()

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set the device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Specify the path to your scenario file.
scenario_folder = pathlib.Path('/dss/dsshome1/05/di38kid/data/scenarios')

# Specify the path to your tower-measurements.json file.
tower_file = pathlib.Path(
    "/dss/dsshome1/05/di38kid/data/paint/WRI1030197-tower-measurements.json"
)

# Specify the following data for each heliostat that you want to include in the scenario:
AA26_heliostat_files_list = [
    (
        "AA26",  #[01]
        pathlib.Path(
            "/dss/dsshome1/05/di38kid/data/paint/selected_20/AA26/Properties/AA26-heliostat-properties.json"
        ),
        pathlib.Path(
            "/dss/dsshome1/05/di38kid/data/paint/selected_20/AA26/Deflectometry/AA26-2021-10-13Z09-34-21Z-deflectometry.h5"
        ),
    ),
]


def generate_scenario(heliostat_files_list: list, name: str, max_epochs_nurbs: 3000):
    """
    Generate the ARTIST-scenario based on the given inputs.
    
    Parameters
    ----------
    heliostat_files_list : list
        A list of tuples specifying heliostat IDs (str), properties files (pathlib.Path) and surface file (pathlib.Path).
    name : str
        The scenario name.
    max_epochs_nurbs : int
        Maximum epochs for nurbs fitting (default: 3000).
    """
    scenario_path = scenario_folder / name

    # Include the power plant configuration.
    power_plant_config, target_area_list_config = (
        paint_loader.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device
        )
    )

    # Include the light source configuration.
    light_source1_config = LightSourceConfig(
        light_source_key="sun_1",
        light_source_type=config_dictionary.sun_key,
        number_of_rays=10,
        distribution_type=config_dictionary.light_source_distribution_is_normal,
        mean=0.0,
        covariance=2 * 4.3681e-06,
    )

    # Create a list of light source configs - in this case only one.
    light_source_list = [light_source1_config]

    # Include the configuration for the list of light sources.
    light_source_list_config = LightSourceListConfig(light_source_list=light_source_list)

    target_area = [
        target_area
        for target_area in target_area_list_config.target_area_list
        if target_area.target_area_key == config_dictionary.target_area_reveicer
    ]

    heliostat_list_config, prototype_config = paint_loader.extract_paint_heliostats(
        heliostat_and_deflectometry_paths=heliostat_files_list,
        power_plant_position=power_plant_config.power_plant_position,
        aim_point=target_area[0].center,
        max_epochs_for_surface_training=max_epochs_nurbs,
        device=device,
    )
    """Generate the scenario given the defined parameters."""
    scenario_generator = ScenarioGenerator(
        file_path=scenario_path,
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    )
    scenario_generator.generate_scenario()
    

def load_scenario(scenario_path: pathlib.Path):
    """
    Load scenario from the given path location.
    
    Parameters
    ----------
    scenario_path : pathlib.Path
        Path location of the scenario.
        
    Return
    ------
    scenario : Scenario
        The loaded scenario. 
    """
    with h5py.File(scenario_path) as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )
    return scenario


def compute_error_in_reflection_axis(flux_center: torch.Tensor, orientation: torch.Tensor, reflection_ax: torch.Tensor):
    """
    Compute the angular offset in reflection axis and COM-vector.
    """
    vect1 = (flux_center - orientation[:, 3])  # com-vector
    vect2 = reflection_ax  # reflection axis
    m1 = torch.norm(vect1, dim=-1, dtype=torch.float64)
    m2 = torch.norm(vect2, dim=-1, dtype=torch.float64)
    dot_product = torch.sum(vect1 * vect2, dim=-1)
    cos_sim = (dot_product + 1e-12) / (m1 * m2 + 1e-12)
    angle_rad = torch.acos(torch.clamp(cos_sim, min= -1.0, max= 1.0)).abs()
    error_in_mrad = angle_rad * 1000
    return error_in_mrad


def process_flux_image(bitmap: torch.Tensor, scenario: Scenario, target_area, incident_ray_direction: torch.Tensor, 
                       save_path: pathlib.Path, cross_legend: bool=False, apply_blocking: bool=False):
    """
    Compute center-crosses for COM and reflection axis intersect point. 
    Generate flux image with center-crosses and save it.
    """       
    orientations = scenario.heliostat_field.rigid_body_kinematic.orientations
    surface_normal = orientations[0, 0:4, 2]
    reflected_ray = raytracing_utils.reflect(incident_ray_direction, surface_normal)
    ideal_center_point, t = calculate_intersection(orientations[0, 0:4, 3], reflected_ray, target_area.center, target_area.normal_vector)
    print(orientations[0, 0:4, 3].tolist())
    
    if apply_blocking:
        sigma = compute_sigma_pixels(t.item(), target_area.plane_u, bitmap.shape[0])
        bitmap = apply_artificial_blocking(bitmap, threshold=0.15, sigma=sigma)
    
    center_of_mass = utils.get_center_of_mass(bitmap=bitmap,
                                              target_center=target_area.center,
                                              plane_e=target_area.plane_e,
                                              plane_u=target_area.plane_u,
                                              threshold=0.0,
                                              device=device
                                              )
    com_indices = get_bitmap_indices_from_center_coordinates(bitmap, center_of_mass, target_area.center, target_area.plane_e, target_area.plane_u, device)

    ideal_indices = get_bitmap_indices_from_center_coordinates(bitmap, ideal_center_point, target_area.center, target_area.plane_e, target_area.plane_u, device)
    angl_offset = compute_error_in_reflection_axis(center_of_mass, orientations[0], reflected_ray).item()
    save_bitmap_with_center_crosses(bitmap, [com_indices, ideal_indices], save_path, cross_legend=cross_legend, offset=angl_offset)
    print(compute_error_in_reflection_axis(center_of_mass, orientations[0], reflected_ray).item()) 


"""Generate scenario with planar surface data."""
planar_scenario_name = 'scenario_planar_AA26'
# generate_scenario(AA26_heliostat_files_list, planar_scenario_name, 0)

"""Generate scenario with deflectometry surface data."""
defl_scenario_name = 'scenario_3000epochs_AA26'
# generate_scenario(AA26_heliostat_files_list, defl_scenario_name, 3000)

"""Load both scenarios with planar and deflectometry surfaces."""
# planar_scenario = load_scenario(scenario_folder / (planar_scenario_name + '.h5'))
# defl_scenario = load_scenario(scenario_folder / (defl_scenario_name + '.h5'))

# Use south sun position
incident_ray_direction_south = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

# Generate image with deflectometry measurements and actual heliostat position
# target_area0 = defl_scenario.get_target_area("solar_tower_juelich_upper")
# bitmap0 = normalize_images(align_and_raytrace(defl_scenario, incident_ray_direction_south.unsqueeze(0), [target_area0], aim_points=target_area0.center.unsqueeze(0))).squeeze(0)
save_dir = pathlib.Path("/dss/dsshome1/05/di38kid/data/results/plots/surface_errors")
# process_flux_image(bitmap0, defl_scenario, target_area0, incident_ray_direction_south, save_dir / "AA26_flux_2.png")

# Generate image with deflectometry measurements and extreme coordinate position
# original_e_pos = defl_scenario.heliostat_field.all_heliostat_positions[0, 0].item()
# defl_scenario.heliostat_field.all_heliostat_positions[0, 0] -= 20.0
# bitmap1 = normalize_images(align_and_raytrace(defl_scenario, incident_ray_direction_south.unsqueeze(0), [target_area0], aim_points=target_area0.center.unsqueeze(0))).squeeze(0)
# process_flux_image(bitmap1, defl_scenario, target_area0, incident_ray_direction_south, save_dir / "AA26_flux_3.png")

# Generate image with deflectometry measurements and heliostat position right in front of the target area
# defl_scenario.heliostat_field.all_heliostat_positions[0, 0] = (target_area0.center[0].item() - original_e_pos) / 2 + original_e_pos
# bitmap2 = normalize_images(align_and_raytrace(defl_scenario, incident_ray_direction_south.unsqueeze(0), [target_area0], aim_points=target_area0.center.unsqueeze(0))).squeeze(0)
#process_flux_image(bitmap2, defl_scenario, target_area0, incident_ray_direction_south, save_dir / "AA26_flux_1.png")

# Generate image with deflectometry measurements and heliostat position right in front of the target area
# defl_scenario.heliostat_field.all_heliostat_positions[0, 0] += target_area0.center[0].item() + 40
# bitmap2 = normalize_images(align_and_raytrace(defl_scenario, incident_ray_direction_south.unsqueeze(0), [target_area0], aim_points=target_area0.center.unsqueeze(0))).squeeze(0)
# process_flux_image(bitmap2, defl_scenario, target_area0, incident_ray_direction_south, save_dir / "AA26_flux_4.png")

# Geneate images for blocking / shading at differnt distances to the tower (north-direction)
aa39_scenario = load_scenario('/dss/dsshome1/05/di38kid/data/scenarios/250129-1347_scenario_AA39.h5')
aa39_scenario.heliostat_field.all_surface_points[0, :, 2] = 0.0 
aa39_scenario.heliostat_field.all_surface_normals[0, : , :] = torch.tensor([0, 0, 1, 0])
target_area0 = aa39_scenario.get_target_area("solar_tower_juelich_lower")

# Original position
aa39_scenario.heliostat_field.all_heliostat_positions[0, 0] = target_area0.center[0].item()
bitmap = normalize_images(align_and_raytrace(aa39_scenario, incident_ray_direction_south.unsqueeze(0), [target_area0], aim_points=target_area0.center.unsqueeze(0))).squeeze(0)
process_flux_image(bitmap, aa39_scenario, target_area0, incident_ray_direction_south, save_dir / 'AA39_flux_0.png', cross_legend=True, apply_blocking=True)

# Original position
aa39_scenario.heliostat_field.all_heliostat_positions[0, 1] += 50
bitmap = normalize_images(align_and_raytrace(aa39_scenario, incident_ray_direction_south.unsqueeze(0), [target_area0], aim_points=target_area0.center.unsqueeze(0))).squeeze(0)
process_flux_image(bitmap, aa39_scenario, target_area0, incident_ray_direction_south, save_dir / 'AA39_flux_1.png', cross_legend=False, apply_blocking=True)

# Original position
aa39_scenario.heliostat_field.all_heliostat_positions[0, 1] += 50
bitmap = normalize_images(align_and_raytrace(aa39_scenario, incident_ray_direction_south.unsqueeze(0), [target_area0], aim_points=target_area0.center.unsqueeze(0))).squeeze(0)
process_flux_image(bitmap, aa39_scenario, target_area0, incident_ray_direction_south, save_dir / 'AA39_flux_2.png', cross_legend=False, apply_blocking=True)

# Original position
aa39_scenario.heliostat_field.all_heliostat_positions[0, 1] += 50
bitmap = normalize_images(align_and_raytrace(aa39_scenario, incident_ray_direction_south.unsqueeze(0), [target_area0], aim_points=target_area0.center.unsqueeze(0))).squeeze(0)
process_flux_image(bitmap, aa39_scenario, target_area0, incident_ray_direction_south, save_dir / 'AA39_flux_3.png', cross_legend=False, apply_blocking=True)


sys.exit()

defl_scenario.heliostat_field.all_heliostat_positions[0, 0] = original_e_pos
defl_scenario.heliostat_field.all_surface_points[0, :, 2] = 0.0 
defl_scenario.heliostat_field.all_surface_normals[0, : , :] = torch.tensor([0, 0, 1, 0])
bitmap3 = normalize_images(align_and_raytrace(defl_scenario, incident_ray_direction_south.unsqueeze(0), [target_area0], aim_points=target_area0.center.unsqueeze(0))).squeeze(0)
process_flux_image(bitmap3, defl_scenario, target_area0, incident_ray_direction_south, save_dir / "AA26_flux_0.png", cross_legend=True)

sys.exit()

planar_scenario.heliostat_field.all_heliostat_positions[0, 1] = 50
planar_scenario.heliostat_field.all_heliostat_positions[0, 2] = target_area.center[2].item()
planar_scenario.heliostat_field.all_surface_points[0, :, 2] = 0.0 
planar_scenario.heliostat_field.all_surface_normals[0, : , :] = torch.tensor([0, 0, 1, 0])

import copy
# print(aa39_scenario.heliostat_field.all_heliostat_positions[0, 0].item())
# print(aa39_scenario_copy.heliostat_field.all_heliostat_positions[0, 0].item())
target_area1 = aa39_scenario_copy.get_target_area("solar_tower_juelich_lower")

# defl_scenario0 = copy.deepcopy(defl_scenario)
# target_area1 = defl_scenario0.get_target_area("solar_tower_juelich_lower")
# target_area2 = defl_scenario.get_target_area("solar_tower_juelich_upper")
# defl_scenario.heliostat_field.all_heliostat_positions[0, 0] = 0.0

"""Align to target center and raytrace."""

# Use upper target in solar tower Jülich


bitmap0 = normalize_images(align_and_raytrace(aa39_scenario, incident_ray_direction_south.unsqueeze(0), [target_area0], aim_points=target_area0.center.unsqueeze(0))).squeeze(0)
bitmap1 = normalize_images(align_and_raytrace(aa39_scenario_copy, incident_ray_direction_south.unsqueeze(0), [target_area1], aim_points=target_area1.center.unsqueeze(0))).squeeze(0)
# bitmap2 = normalize_images(align_and_raytrace(defl_scenario, incident_ray_direction_south.unsqueeze(0), [target_area2], aim_points=target_area2.center.unsqueeze(0))).squeeze(0)

save_dir = pathlib.Path("/dss/dsshome1/05/di38kid/data/results/plots/surface_errors")
#process_flux_image(bitmap0, aa39_scenario, target_area0, incident_ray_direction_south, save_dir / "AA39_flux_1.png")
# process_flux_image(bitmap1, aa39_scenario_copy, target_area1, incident_ray_direction_south, save_dir / "AA39_flux_0.png")
# process_flux_image(bitmap1, defl_scenario, target_area2, incident_ray_direction_south, save_dir / "flux_1.png")
# process_flux_image(bitmap2, defl_scenario, target_area2, incident_ray_direction_south, save_dir / "flux_3.png")

