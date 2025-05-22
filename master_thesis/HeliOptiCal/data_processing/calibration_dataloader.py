import json
import os
import sys
import logging
import torch
import torchvision.transforms as transforms
import pathlib
import random
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional, Any, Literal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import paint.util.paint_mappings as mappings
from paint.data.dataset_splits import DatasetSplitter

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, repo_path) 
from HeliOptiCal.utils import my_config_dict
import HeliOptiCal.utils.util_dataset as utils_dataset
import HeliOptiCal.utils.my_config_dict as my_config_dict

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
# A logger for the calibration dataset.

class CalibrationDataLoader:
    """
    A class for managing heliostat calibration data.

    Methods
    -------
    __init__(...)
        Initializes the loader and loads calibration metadata and optionally flux images.
    load_properties_from_file(properties_file, device=None)
        Extracts and returns data from a single calibration JSON file
    get_field_batch(helio_and_calib_ids=None, device=None)
        Constructs a batch of field-level data samples for simulation or inference tasks.
    """
    def __init__(
            self,
            data_directory: Union[str, Path],
            heliostats_to_load: List[str],
            power_plant_position: torch.Tensor,
            load_flux_images: bool = True,
            preload_flux_images: bool = False,
            is_simulated_data: bool = False,
            properties_file_ends_with: Optional[str] = '-calibration-properties.json',
            flux_file_ends_with: Optional[str] = '-flux.png',
            device: Union[torch.device, str] = "cuda"
    ):
        """
        Initialize the CalibrationDataLoader with a data directory.

        Parameters
        ----------
        data_directory : Union[str, Path]
            Path to the directory containing calibration data files with 'heliostat_id' as folders.
        heliostats_to_load : List[str]: 
            List of heliostat IDs to load calibration data for.
        power_plant_position : torch.Tensor 
            The position of the power plant in the world coordinate system.
        load_flux_images : bool 
            Whether to load flux images from PNG files (default: True).
        preload_flux_images : bool
            Whether to pre-load flux images (default: False).
        is_simulated_data : bool
            Whether the data was generated using the Raytracer (default: False).
        properties_file_ends_with : str 
            File ending for calibration properties JSON files (default: '-calibration-properties.json').
        flux_file_ends_with : str 
            File ending for flux images PNG files (default: '-flux.png').
        device : Union[torch.device, str] 
            Device to load data tensors onto (default: 'cuda').
        """
        self.data_dir = data_directory
        self.heliostats_to_load = heliostats_to_load
        self.power_plant_position = power_plant_position
        self.load_flux_images = load_flux_images
        self.preload_flux_images = preload_flux_images
        self.device = torch.device(device)
        
        # Will influence how properties data is loaded 
        self.is_simulated_data = is_simulated_data  # False: Data was measured (`PAINT`)
        
        # File endings for calibration properties and flux images
        self.properties_file_ends_with = properties_file_ends_with
        self.flux_file_ends_with = flux_file_ends_with

        # Storage dictionaries for each data type
        self.calibration_ids: Dict[str, list] = {}
        self.corrupt_data_ids: Dict[str, set] = {}
        self.sun_azimuths: Dict[int, float] = {}
        self.sun_elevations: Dict[int, float] = {}
        self.sun_positions: Dict[int, float] = {}
        self.flux_centers: Dict[int, torch.Tensor] = {}
        self.ideal_flux_centers: Dict[int, torch.Tensor] = {}
        self.motor_positions: Dict[int, torch.Tensor] = {}
        self.incident_rays: Dict[int, torch.Tensor] = {}
        self.receiver_targets: Dict[int, str] = {}
        self.flux_images: Dict[int, torch.Tensor] = {}
        
        # Storage dictionaries for paths to properties files and flux image files
        self.properties_file_paths: Dict[int, Path] = {}
        self.flux_file_paths: Dict[int, Path] = {}

        # Load the data
        self._load_data()
        
    def load_properties_from_file(self, properties_file: Union[str, Path], device: Optional[Union[torch.device, str]] = None):
        """
        Load data from properties file if available.
        """
        # Select device if None was given
        device = self.device if device is None else device
        
        already_in_enu_4d = False  # ``PAINT`` data will be in wg84 coordinates
        
        # Extract calibration ID from digits in file name
        calibration_id = int(''.join(filter(str.isdigit, Path(properties_file).name)))
        
        # Check that calibration ID is unique
        if calibration_id in set(id for hel_ids in self.calibration_ids.values() for id in hel_ids):
            raise ValueError(f"Detected recurrence for calibration ID {calibration_id} in dataset! "\
                             "\nCalibration IDs must be unique.")
        
        # Extract data from JSON properties file       
        (
            calibration_target_name,
            flux_center,
            sun_azimuth,
            sun_elevation,
            sun_position,
            motor_positions,
        ) = utils_dataset.extract_paint_calibration_data(
            calibration_properties_path=Path(properties_file),
            power_plant_position=self.power_plant_position,
            already_in_enu_4d=self.is_simulated_data,
            device=self.device
        )
        
        return (
            calibration_id,
            flux_center,
            calibration_target_name,
            sun_azimuth,
            sun_elevation,
            sun_position,
            motor_positions,
        )
        
    def _load_properties_data_for_heliostat(self, heliostat_id: str, device: Optional[Union[torch.device, str]] = None):
        """
        Return all calibration IDs for selected Heliostat with available data files (JSON and PNG)
        that can be found in the calibration directory.
        """
        # Select device if None was given
        device = torch.device(self.device if device is None else device)
        
        # Define data directory for this heliostat_id
        calibration_directory = (Path(self.data_dir) / f'{heliostat_id}/Calibration')         
            
        # Check if data directory exists
        if not os.path.exists(calibration_directory):
            raise FileNotFoundError(f"Calibration directory not at: {calibration_directory}")
        else:
            log.info(f"Loading data for Heliostat {heliostat_id} from: {calibration_directory}")
        
        # Load all calibration IDs which have data files
        # At least the properties-file must exist for valid calibration data
        properties_files = [os.path.join(calibration_directory, f)
                            for f in os.listdir(calibration_directory) 
                            if f.endswith(self.properties_file_ends_with)]
        
        # non-existent for ``PAINT`` data, ie. measured data
        load_ideal_flux_center = True if self.is_simulated_data else False 
        
        self.calibration_ids[heliostat_id] = []
        nan_flux_centers = []
        
        for properties_file in properties_files:
            (
                calibration_id,
                flux_center,
                calibration_target_name,
                sun_azimuth,
                sun_elevation,
                sun_position,
                motor_positions,
            ) = self.load_properties_from_file(properties_file, device=device)
            
            # Load ideal flux center from properties file if available
            if load_ideal_flux_center:
                with open(properties_file, 'r') as f:
                    calibration_dict = json.load(f)
                
                ideal_flux_center = torch.tensor(
                    calibration_dict[my_config_dict.ideal_focal_spot_enu_4d],
                    dtype=torch.float64,
                    device=device,
                )
                self.ideal_flux_centers[calibration_id] = ideal_flux_center

            # Save data to storage dictionaries
            self.calibration_ids[heliostat_id].append(calibration_id)
            
            # Save data to storage dictionaries
            self.sun_azimuths[calibration_id] = sun_azimuth
            self.sun_elevations[calibration_id] = sun_elevation
            self.sun_positions[calibration_id] = sun_position
            self.flux_centers[calibration_id] = flux_center          
            self.motor_positions[calibration_id] = motor_positions
            self.incident_rays[calibration_id] = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(device) - sun_position.to(device)
            self.receiver_targets[calibration_id] = calibration_target_name
            
            # Save properties file path
            self.properties_file_paths[calibration_id] = Path(properties_file)
            
            if torch.isnan(flux_center).any():
                nan_flux_centers.append(calibration_id)
        
        log.info(f"{heliostat_id}: Found {len(properties_files)} calibration properties files.")
        if len(nan_flux_centers) > 0:
            log.warning(f"However, some have invalid flux centers {nan_flux_centers}")
            self.corrupt_data_ids[heliostat_id] = set(nan_flux_centers)
    
    def _load_flux_image(self, flux_image_path, device: Optional[Union[torch.device, str]] = None):
        """
        Load a flux image and return as Tensor.
        """
        # Select device if None was given
        device = torch.device(self.device if device is None else device)
        
        flux_image = Image.open(flux_image_path)
        
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor()  # if you want it as a PyTorch tensor
        ])
        flux_image = transform(flux_image).squeeze(0)
        
        return flux_image
    
    def _load_flux_images_for_heliostat(self, heliostat_id: str):
        """
        Load the paths to the flux images for the given Heliostat.
        Store flux images, if option pre-load was chosen.
        """
        missing_flux_images = []
        # Iterate over the Heliostat's calibration IDs
        for calibration_id in self.calibration_ids[heliostat_id]:
            # Get the path of the flux image path by replacing file endings
            properties_file_path = str(self.properties_file_paths[calibration_id])
            flux_image_path = Path(
                properties_file_path.replace(self.properties_file_ends_with,
                                             self.flux_file_ends_with)
                )
            
            # Raise Error if the flux image cannot be found, else add to paths
            if not os.path.exists(flux_image_path):
                self.flux_file_paths[calibration_id] = None
                missing_flux_images.append(calibration_id)
                
            else:
                self.flux_file_paths[calibration_id] = flux_image_path
                # Load flux images if pre-load is True
                if self.preload_flux_images:
                    try:
                        self.flux_images[calibration_id] = self._load_flux_image(flux_image_path)
                    except (MemoryError, RuntimeError) as e:
                        raise MemoryLimitExceeded("Memory limit exceeded during preload. "\
                                                  "Consider setting preload_flux_images=False.") from e
                    
        if len(missing_flux_images) == 0:
            log.info(f"All flux images for {heliostat_id} were found.")
        else:
            log.warning(f"Some flux images were not found: {missing_flux_images}")
            self.corrupt_data_ids[heliostat_id] = set(missing_flux_images)
    
    def _load_data(self): 
        """
        Load all available calibration data for the chosen heliostats.
        """   
        heliostats = self.heliostats_to_load
        log.info(f"Finding available calibration data for Heliostats {heliostats}...")
        
        calibration_directories = {}
        
        for heliostat_id in heliostats:
            calibration_directory = (Path(self.data_dir) / f'{heliostat_id}/Calibration')
            if not os.path.exists(calibration_directory):
                raise FileNotFoundError(f"Calibration folder not found at path: "
                                        f"{calibration_directory}")
            else:
                calibration_directories[heliostat_id] = calibration_directory
                self._load_properties_data_for_heliostat(heliostat_id)
                
                if self.load_flux_images:
                    self._load_flux_images_for_heliostat(heliostat_id)

        log.info(f"Initial loading of calibration data complete.")
    
    def get_field_batch(self, 
                        helio_and_calib_ids: Optional[Dict[str, List[int]]] = None, 
                        device: Optional[Union[torch.device, str]] = None):
        """
        Get a data batch for the heliostat field, which can be used for parallel Field-Raytracing
        with unique sun positions per Heliostat and different receiver targets.

        Parameters
        ----------
        heliostat_and_calib_ids : Optional[Dict[str, List[int]]]
            Selected Heliostats as keys and calibration IDs as values.
            Leave at default to load the whole data set for the field (default is None).
        device : Optional[Union[torch.device, str]]
            Select device. Use dataset default device if None (default is None).
        
        Returns
        -------
        field_batch : List[Dict]
            A list of data samples for the heliostat field. Length of list is batch size.
            Each Heliostat field sample has keys to data.
        """
        # Select device if None was given
        device = torch.device(self.device if device is None else device)
        
        # Check if Heliostats are in this dataset
        not_in_field = set(hel_id for hel_id in helio_and_calib_ids 
                        if hel_id not in self.calibration_ids)
        if not_in_field:
            raise KeyError(f"No data was loaded to the dataset for your Heliostats: {not_in_field}")
        
        # Check if all requested IDs are available
        missing_ids = (set(batch_id for ids in helio_and_calib_ids.values() for batch_id in ids)
                       - set(cal_id for ids in self.calibration_ids.values() for cal_id in ids))
        if missing_ids:
            raise ValueError(f"Calibration IDs not found in the loaded dataset: {missing_ids}")
        
        # Check if any requested IDs were registered with corrupt data
        corrupt_ids = (set(batch_id for ids in helio_and_calib_ids.values() for batch_id in ids
                           if batch_id in set(cal_id for ids in self.corrupt_data_ids.values() for cal_id in ids)))
        if corrupt_ids:
            info.warning(f'The requested data batch may contain corrupt data: {corrupt_ids}')

        # Load the whole dataset if None was given
        if len(helio_and_calib_ids) is None:
            helio_and_calib_ids = self.calibration_ids
        
        # Length of samples of calibration data per heliostat
        sample_length = len(next(iter(helio_and_calib_ids.values())))
        
        field_batch = []
        # Iterate over the number of samples for the Heliostat field and collect batch data
        for sample in range(sample_length):
            # Sample one calibration ID per heliostat for current sample index
            sample_ids = [int(cal_ids[sample]) for cal_ids in helio_and_calib_ids.values()]
            
            sample_flux_centers = torch.stack([self.flux_centers[id] for id in sample_ids]).to(device)
            sample_motor_positions = torch.stack([self.motor_positions[id] for id in sample_ids]).to(device)
            sample_incident_rays = torch.stack([self.incident_rays[id] for id in sample_ids]).to(device)
            sample_receiver_targets = [self.receiver_targets[id] for id in sample_ids]

            sample_data = {
                my_config_dict.field_sample_calibration_ids: sample_ids,
                my_config_dict.field_sample_sun_azimuths: [self.sun_azimuths[id] for id in sample_ids],
                my_config_dict.field_sample_sun_elevations: [self.sun_elevations[id] for id in sample_ids],
                my_config_dict.field_sample_flux_centers: sample_flux_centers,
                my_config_dict.field_sample_motor_positions: sample_motor_positions,
                my_config_dict.field_sample_incident_rays: sample_incident_rays,
                my_config_dict.field_sample_target_names: sample_receiver_targets
                }

            if self.is_simulated_data:
                sample_ideal_flux_centers = torch.stack([self.ideal_flux_centers[id] for id in sample_ids]).to(device)
                sample_data[my_config_dict.field_sample_ideal_flux_centers] = sample_ideal_flux_centers
            
            if self.load_flux_images:
                if self.preload_flux_images:
                    sample_flux_images = torch.stack([self.flux_images[id] for id in sample_ids]).to(device)
                else:
                    sample_flux_images = torch.stack([self._load_flux_image(self.flux_file_paths[id]) for id in sample_ids]).to(device)
                    
                sample_data[my_config_dict.field_sample_flux_images] = sample_flux_images
                
            field_batch.append(sample_data)
        return field_batch
