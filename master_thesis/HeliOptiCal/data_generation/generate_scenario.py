import pathlib
import os
import sys
import time

import torch

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 
from artist.util import config_dictionary, paint_loader, set_logger_config
from artist.util.configuration_classes import (
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.util.scenario_generator import ScenarioGenerator


def build_heliostat_file_list(central_data_dir: str):
    """
    Scan a central heliostat data directory and collect file paths for each valid heliostat.

    For each heliostat folder (name pattern: two uppercase letters + two digits), this function:
    - Retrieves the first JSON file in the 'Properties' subdirectory.
    - Finds the latest alphabetically sorted deflectometry .h5 file from the 'Deflectometry' subdirectory
      (excluding files that contain the substring 'filled').

    Parameters
    ----------
    central_data_dir : str
        Path to the root directory containing one folder per Heliostat.

    Returns
    -------
    List[Tuple[str, pathlib.Path, pathlib.Path]]
        A list of tuples of the form:
        (HeliostatId, heliostat-properties.json path, deflectometry.h5 path)
    """
    base_path = pathlib.Path(central_data_dir)
    heliostat_files_list = []
    
    for heliostat_dir in sorted(base_path.iterdir()):
        if not heliostat_dir.is_dir():
            continue
        
        heliostat_id = heliostat_dir.name
        if len(heliostat_id) != 4 or not heliostat_id[:2].isalpha() or not heliostat_id[2:].isdigit():
            continue  # Skip non-matching folder names
        
        # --- Locate Properties file ---
        properties_dir = heliostat_dir / "Properties"
        if not properties_dir.exists():
            continue
        properties_files = list(properties_dir.glob("*.json"))
        if len(properties_files) == 0:
            continue
        properties_file = properties_files[0]  # take first one

        # --- Locate valid Deflectometry file ---
        deflectometry_dir = heliostat_dir / "Deflectometry"
        if not deflectometry_dir.exists():
            continue
        deflectometry_files = [
            f for f in deflectometry_dir.glob("*deflectometry.h5")
            if "filled" not in f.name
        ]
        if len(deflectometry_files) == 0:
            continue
        deflectometry_files.sort()  # sort alphabetically
        deflectometry_file = deflectometry_files[-1]  # take the latest one

        # --- Append entry ---
        heliostat_files_list.append(
            (
                heliostat_id,
                properties_file,
                deflectometry_file,
            )
        )

    return heliostat_files_list


# Set up logger
set_logger_config()

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set the device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Specify the path to your scenario file.
scenario_folder = pathlib.Path('/dss/dsshome1/05/di38kid/data/scenarios')
scenario_name = '20_Heliostats'

# Specify the path to your tower-measurements.json file.
tower_file = pathlib.Path("/dss/dsshome1/05/di38kid/data/paint/WRI1030197-tower-measurements.json")

# A tuple of: (helisotat-name, heliostat-properties.json, deflectometry.h5)
heliostat_files_list = build_heliostat_file_list("/dss/dsshome1/05/di38kid/data/paint/selected_20")
print(f"Found {len(heliostat_files_list)} valid heliostats.")
scenario_path = scenario_folder / f'{scenario_name}.h5'
print(scenario_path)

# This checks to make sure the path you defined is valid and a scenario HDF5 can be saved there.
if not pathlib.Path(scenario_path).parent.is_dir():
    raise FileNotFoundError(
        f"The folder ``{pathlib.Path(scenario_path).parent}`` selected to save the scenario does not exist. "
        "Please create the folder or adjust the file path before running again!"
    )

# Include the power plant configuration.
power_plant_config, target_area_list_config = (
    paint_loader.extract_paint_tower_measurements(tower_measurements_path=tower_file, device=device)
)

# Include the light source configuration.
light_source1_config = LightSourceConfig(light_source_key="sun_1",
                                         light_source_type=config_dictionary.sun_key,
                                         number_of_rays=10,
                                         distribution_type=config_dictionary.light_source_distribution_is_normal,
                                         mean=0.0, covariance=2*4.3681e-06)

# Create a list of light source configs - in this case only one.
light_source_list = [light_source1_config]

# Include the configuration for the list of light sources.
light_source_list_config = LightSourceListConfig(light_source_list=light_source_list)

target_area = [target_area for target_area in target_area_list_config.target_area_list
               if target_area.target_area_key == config_dictionary.target_area_reveicer]

heliostat_list_config, prototype_config = paint_loader.extract_paint_heliostats(
    heliostat_and_deflectometry_paths=heliostat_files_list,
    power_plant_position=power_plant_config.power_plant_position,
    aim_point=target_area[0].center,
    max_epochs_for_surface_training=4000,
    device=device,
)


if __name__ == "__main__":
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
