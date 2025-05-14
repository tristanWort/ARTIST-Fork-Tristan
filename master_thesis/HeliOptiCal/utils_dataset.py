import json
import torch
from pathlib import Path
from typing import Union, Literal, Dict, List

from artist.util import config_dictionary, utils

import my_config_dict


def extract_paint_calibration_data(calibration_properties_path: Union[str, Path],
                                   power_plant_position: torch.Tensor,
                                   already_in_enu_4d: bool = False,
                                   device: Union[torch.device, str] = "cuda"
                                   ):
    """
    Extract calibration data from ``PAINT`` calibration files.

    Parameters
    ----------
    calibration_properties_path : Union[str, Path]
        The path to the calibration properties file.
    power_plant_position : torch.Tensor
        The position of the power plant in latitude, longitude and elevation.
    already_in_enu_4d : bool
        Whether the center coordinatese are given in ENU 4D (default is False).
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    str
        The name of the calibration target.
    torch.Tensor
        The calibration focal spot center in ENU 4D.
    torch.Tensor
        The sun azimuth.
    torch.Tensor
        The sun elevation.
    torch.Tensor
        The sun position in ENU 4D.
    torch.Tensor
        The motor positions.
    """
    device = torch.device(device)
    
    with open(calibration_properties_path, "r") as file:
        calibration_dict = json.load(file)
        
    calibration_target_name = calibration_dict[config_dictionary.paint_calibration_target]
    
    if already_in_enu_4d: 
        center_calibration_image = torch.tensor(
            calibration_dict[my_config_dict.focal_spot_enu_4d],
            dtype=torch.float64,
            device=device
            )
        
    else:
        # Convert coordinates to local ENU in 4D
        try:
            center_calibration_image = utils.convert_wgs84_coordinates_to_local_enu(
                torch.tensor(
                    calibration_dict[config_dictionary.paint_focal_spot][
                        config_dictionary.paint_utis
                    ],
                    dtype=torch.float64,
                    device=device,
                ),
                power_plant_position,
                device=device,
            )
            center_calibration_image = utils.convert_3d_direction_to_4d_format(
                center_calibration_image, device=device
            )
            
        except KeyError:
            center_calibration_image = torch.full((4,), float('nan'), device=device)
            
    sun_azimuth = torch.tensor(
        calibration_dict[config_dictionary.paint_sun_azimuth], device=device
    )
    sun_elevation = torch.tensor(
        calibration_dict[config_dictionary.paint_sun_elevation], device=device
    )
    sun_position_enu = utils.convert_3d_point_to_4d_format(
        utils.azimuth_elevation_to_enu(
            sun_azimuth, sun_elevation, degree=True, device=device
        ),
        device=device,
    )
    motor_positions = torch.tensor(
        [
            calibration_dict[config_dictionary.paint_motor_positions][
                config_dictionary.paint_first_axis
            ],
            calibration_dict[config_dictionary.paint_motor_positions][
                config_dictionary.paint_second_axis
            ],
        ],
        device=device,
    )
    
    return (
        calibration_target_name,
        center_calibration_image,
        sun_azimuth,
        sun_elevation,
        sun_position_enu,
        motor_positions,
    )

