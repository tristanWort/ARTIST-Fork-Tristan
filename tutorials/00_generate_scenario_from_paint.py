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

# Set up logger
set_logger_config()

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Set the device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Specify the path to your scenario file.
scenario_folder = pathlib.Path('/dss/dsshome1/05/di38kid/data/scenarios')
# scneario_name = 

# Specify the path to your tower-measurements.json file.
tower_file = pathlib.Path(
    "/dss/dsshome1/05/di38kid/data/paint/WRI1030197-tower-measurements.json"
)

# Specify the following data for each heliostat that you want to include in the scenario:
# A tuple of: (helisotat-name, heliostat-properties.json, deflectometry.h5)
heliostat_files_list = [
    (
        "AA39",  #[01]
        pathlib.Path(
            "/dss/dsshome1/05/di38kid/data/paint/AA39/Properties/AA39-heliostat-properties.json"
        ),
        pathlib.Path(
            "/dss/dsshome1/05/di38kid/data/paint/AA39/Deflectometry/AA39-2023-09-18Z08-49-09Z-deflectometry.h5"
        ),
    ),
    (
        "AC27",  #[02]
        pathlib.Path(
            "/dss/dsshome1/05/di38kid/data/paint/AC27/Properties/AC27-heliostat-properties.json"
        ),
        pathlib.Path(
            "/dss/dsshome1/05/di38kid/data/paint/AC27/Deflectometry/AC27-2021-10-13Z09-50-00Z-deflectometry.h5"
        ),
    ),
    (
        "AD43",  #[03]
        pathlib.Path(
            "/dss/dsshome1/05/di38kid/data/paint/AD43/Properties/AD43-heliostat-properties.json"
        ),
        pathlib.Path(
            "/dss/dsshome1/05/di38kid/data/paint/AD43/Deflectometry/AD43-2023-09-18Z10-44-44Z-deflectometry.h5"
        ),
    ),
    (    "AM35",  #[04]
        pathlib.Path(
            "/dss/dsshome1/05/di38kid/data/paint/AM35/Properties/AM35-heliostat-properties.json"
        ),
        pathlib.Path(
            "/dss/dsshome1/05/di38kid/data/paint/AM35/Deflectometry/AM35-2021-10-13Z11-27-47Z-deflectometry.h5"
        ),
    ),
    (    "BB72",  #[05]
        pathlib.Path(
            "/dss/dsshome1/05/di38kid/data/paint/BB72/Properties/BB72-heliostat-properties.json"
        ),
        pathlib.Path(
            "/dss/dsshome1/05/di38kid/data/paint/BB72/Deflectometry/BB72-2407-11-10Z02-36-00Z-deflectometry.h5"
        ),
    ),
    (    "BG24",  #[06]
        pathlib.Path(
            "/dss/dsshome1/05/di38kid/data/paint/BG24/Properties/BG24-heliostat-properties.json"
        ),
        pathlib.Path(
            "/dss/dsshome1/05/di38kid/data/paint/BG24/Deflectometry/BG24-2016-05-12Z07-43-52Z-deflectometry.h5"
        ),
    ),
    # (
    #     "name2",
    #     pathlib.Path(
    #         "please/insert/the/path/to/the/heliostat/properties/here/heliostat_properties.json"
    #     ),
    #     pathlib.Path(
    #         "please/insert/the/path/to/the/deflectometry/data/here/deflectometry.h5"
    #     ),
    # ),
    # ... Include as many as you want, but at least one!
]

print('save sceanio as:')
scenario_path = scenario_folder / 'scenario_7heliostats.h5'
print(scenario_path)

# This checks to make sure the path you defined is valid and a scenario HDF5 can be saved there.
if not pathlib.Path(scenario_path).parent.is_dir():
    raise FileNotFoundError(
        f"The folder ``{pathlib.Path(scenario_path).parent}`` selected to save the scenario does not exist. "
        "Please create the folder or adjust the file path before running again!"
    )

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
    max_epochs_for_surface_training=3000,
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
