import json
import os
import logging
import torch
import torchvision.transforms as transforms
import pathlib
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional, Any, Literal
from sklearn.cluster import KMeans

from artist.util.paint_loader import  extract_paint_calibration_data

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
# A logger for the calibration dataset.

class CalibrationDataLoader:
    """
    A class for managing heliostat calibration data.

    This class stores and provides access to calibration data including:
    - Flux spot center of weight (4D coordinates)
    - Motor positions (2D)
    - Incident ray directions (4D)
    - Receiver target names
    - Flux image bitmaps (256x256)
    - Calibration IDs

    The data can be loaded from a directory containing JSON and JPG files,
    and can be retrieved by specifying calibration IDs.
    """

    def __init__(
            self,
            data_dir: Union[str, pathlib.Path],
            power_plant_position: torch.Tensor,
            calibration_ids: Optional[List[str]] = [],
            load_flux_images: bool = True,
            properties_file_ends_with: Optional[str] = '-calibration-properties.json',
            flux_file_ends_with: Optional[str] = '-flux.png',
            device: Union[torch.device, str] = "cuda"
    ):
        """
        Initialize the CalibrationDataLoader with a data directory.

        Args:
            data_dir (str, pathlib.Path): Path to the directory containing calibration data files
            calibration_ids (List[str]): List of calibration IDs to load (default: None, which loads all IDs
            in the directory)
        """
        self.data_dir = data_dir
        self.power_plant_position = power_plant_position

        # Storage dictionaries for each data type
        self.sun_azimuths: Dict[str, float] = {}
        self.sun_elevations: Dict[str, float] = {}
        self.flux_centers: Dict[str, torch.Tensor] = {}  # shape [4,]
        self.motor_positions: Dict[str, torch.Tensor] = {}  # shape [2,]
        self.incident_rays: Dict[str, torch.Tensor] = {}  # shape [4,]
        self.receiver_targets: Dict[str, str] = {}
        self.flux_images: Dict[str, torch.Tensor] = {}  # shape [256, 256]

        # List of all calibration IDs
        self.calibration_ids: List[str] = calibration_ids

        self.load_flux_images = load_flux_images

        # File endings for calibration properties and flux images
        self.properties_file_ends_with = properties_file_ends_with
        self.flux_file_ends_with = flux_file_ends_with

        self.device = device

        # Load the data (you'll implement your custom loading logic here)
        self._load_data()

    def _load_data(self):
        """
        Load calibration data from JSON and PNG files in the data directory.

        This method should be customized to implement your specific data loading logic.
        """

        #Todo: Ask Claude if this is good code.
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Calibration folder not found at path: {self.data_dir}")

        log.info(f"Loading calibration data from: {self.data_dir}") 
        if self.calibration_ids is None:  # load all calibration json files in the folder.
            properties_files = [os.path.join(self.data_dir, f)
                                for f in os.listdir(self.data_dir)
                                if f.endswith(self.properties_file_ends_with)]
        else:  # only load the calibration json files which were specified.
            properties_files = []
            for calibration_id in self.calibration_ids:
                try:
                    properties_files.append(os.path.join(self.data_dir,
                                                         calibration_id +
                                                         self.properties_file_ends_with))
                except FileNotFoundError:
                    log.warning(f"Calibration json file not found at: "
                                f"{calibration_id + self.properties_file_ends_with}.")

        self.calibration_ids = []
        for properties_file in properties_files:
            calibration_id = ''.join(filter(str.isdigit, Path(properties_file).name))
            self.calibration_ids.append(calibration_id)

            with open(properties_file, 'r') as file:
                calibration_data = json.load(file)
            self.sun_azimuths[calibration_id] = calibration_data['sun_azimuth']
            self.sun_elevations[calibration_id] = calibration_data['sun_elevation']

            (
                calibration_target_name,
                spot_centers,
                incident_ray_direction,
                motor_positions,
            ) = extract_paint_calibration_data(
                calibration_properties_path=Path(properties_file),
                power_plant_position=self.power_plant_position,
                device=self.device
            )
            self.flux_centers[calibration_id] = spot_centers.to(self.device)
            self.motor_positions[calibration_id] = motor_positions.to(self.device)
            self.incident_rays[calibration_id] = incident_ray_direction.to(self.device)
            self.receiver_targets[calibration_id] = calibration_target_name

            if self.load_flux_images:
                flux_file = properties_file.replace(self.properties_file_ends_with,
                                                    self.flux_file_ends_with)
                if os.path.exists(flux_file):
                    flux_img = Image.open(flux_file)
                    flux_img = (transforms.ToTensor()(flux_img)).squeeze(0)
                    self.flux_images[calibration_id] = flux_img.to(self.device)
                else:
                    raise FileNotFoundError(f"No flux image found for at: "
                                            f"{calibration_id + self.flux_file_ends_with}.")
        log.info(f"Calibration data loaded for {len(self.flux_centers)} calibration IDs.")

    def get_calibration_ids(self) -> List[str]:
        """
        Get all available calibration IDs.

        Returns:
            List[str]: List of all calibration IDs
        """
        return self.calibration_ids.copy()

    def get_batch(self, cal_ids: List[str]) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Get a batch of calibration data for the specified calibration IDs.

        Args:
            cal_ids (List[str]): List of calibration IDs to include in the batch

        Returns:
            Dict: A dictionary containing batched data with the following keys:
                - 'flux_centers': Tensor of shape [N, 4]
                - 'motor_positions': Tensor of shape [N, 2]
                - 'incident_rays': Tensor of shape [N, 4]
                - 'receiver_targets': List of N target names
                - 'flux_images': Tensor of shape [N, 256, 256]
                - 'calibration_ids': List of N calibration IDs

        Raises:
            KeyError: If any of the requested calibration IDs are not available
        """
        # Check if all requested IDs are available
        missing_ids = set(cal_ids) - set(self.calibration_ids)
        if missing_ids:
            raise KeyError(f"Calibration IDs not found in CalibrationDataLoader: {missing_ids}")

        # Stack tensors for batch processing
        batch_flux_centers = torch.stack([self.flux_centers[cal_id] for cal_id in cal_ids]).to(self.device)
        batch_motor_positions = torch.stack([self.motor_positions[cal_id] for cal_id in cal_ids]).to(self.device)
        batch_incident_rays = torch.stack([self.incident_rays[cal_id] for cal_id in cal_ids]).to(self.device)
        batch_receiver_targets = [self.receiver_targets[cal_id] for cal_id in cal_ids]

        batch_data = {'cal_ids': cal_ids,
                      'sun_azimuths': [self.sun_azimuths[cal_id] for cal_id in cal_ids],
                      'sun_elevations': [self.sun_elevations[cal_id] for cal_id in cal_ids],
                      'flux_centers': batch_flux_centers,
                      'motor_positions': batch_motor_positions,
                      'incident_rays': batch_incident_rays,
                      'receiver_targets': batch_receiver_targets}

        if self.load_flux_images:
            batch_flux_images = torch.stack([self.flux_images[cal_id] for cal_id in cal_ids]).to(self.device)
            batch_data['flux_images'] = batch_flux_images

        return batch_data

    def get_single_item(self, cal_id: str) -> Dict[str, Any]:
        """
        Get data for a single calibration ID.

        Args:
            cal_id (str): The calibration ID to retrieve

        Returns:
            Dict: A dictionary containing the data for the requested calibration ID

        Raises:
            KeyError: If the requested calibration ID is not available
        """
        if cal_id not in self.calibration_ids:
            raise KeyError(f"Calibration ID not found: {cal_id}")

        batch_data = {'cal_id': cal_id,
                      'sun_azimuth': self.sun_azimuths[cal_id],
                      'sun_elevation': self.sun_elevations[cal_id],
                      'flux_center': self.flux_centers[cal_id],
                      'motor_position': self.motor_positions[cal_id],
                      'incident_ray': self.incident_rays[cal_id],
                      'receiver_target': self.receiver_targets[cal_id]}

        if self.load_flux_images:
            batch_data['flux_image'] = self.flux_images[cal_id]

        return batch_data

    def filter_by_target(self, target_name: str) -> List[str]:
        """
        Filter calibration IDs by receiver target name.

        Args:
            target_name (str): The receiver target name to filter by

        Returns:
            List[str]: List of calibration IDs that match the target name
        """
        return [cal_id for cal_id in self.calibration_ids
                if self.receiver_targets[cal_id] == target_name]

    def random_batch(
            self,
            batch_size: int,
            seed: Optional[int] = None
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Get a random batch of calibration data.

        Args:
            batch_size (int): Number of calibration points to include in the batch
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            Dict: A dictionary containing batched data (same format as get_batch)

        Raises:
            ValueError: If batch_size is larger than the number of available calibration points
        """
        if batch_size > len(self.calibration_ids):
            raise ValueError(
                f"Requested batch size {batch_size} exceeds available data size {len(self.calibration_ids)}")

        if seed is not None:
            torch.manual_seed(seed)

        # Sample random calibration IDs
        sample_ids = random.sample(range(len(self.calibration_ids)), batch_size)
        selected_ids = [self.calibration_ids[i] for i in sample_ids]

        return self.get_batch(selected_ids)

    def split_data(
        self,
        train_valid_test_sizes: Tuple[int,int,int] = (30, 15, 15),
        split_type: Literal['random','kmeans','target'] = 'random',
        n_clusters: Optional[int] = 5,
        seed: Optional[int] = 42
    ) -> Tuple[List[str],List[str],Optional[List[str]]]:
        """
        Split the data into train, validation, and test (optional) sets.

        Args:
            train_valid_test_sizes (Tuple[int,int,int]): 
                Desired target sample lenght in training, validation, and testing batches.
            split_tpye (Literal[str]): Select a method for data splitting.
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            Tuple[List[str], List[str], Optional[List[str]]: Train, validation, and test calibration IDs.
        """
        random.seed(seed)  # for reproduducibility

        train_size = train_valid_test_sizes[0]
        valid_size = train_valid_test_sizes[1]
        test_size = train_valid_test_sizes[2]

        if split_type == 'random':
            total_size = sum(train_valid_test_sizes)
            use_ids = self.random_batch(total_size, seed=seed)['cal_ids']

            train_ids = use_ids[:train_size]
            valid_ids = use_ids[train_size:train_size + valid_size]
            test_ids = use_ids[-test_size:]

            self.plot_splits_sun_positions(train_ids, valid_ids, test_ids)
            return train_ids, valid_ids, test_ids
        
        elif split_type == 'kmeans':
            import numpy as np
            import matplotlib.pyplot as plt
            sun_azimuths_list = list(self.sun_azimuths.values())
            sun_elevations_list = list(self.sun_elevations.values())
            features = np.array([sun_azimuths_list, sun_elevations_list]).T
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            cluster_labels = kmeans.fit_predict(features)

            plt.scatter(features[:, 0], features[:, 1], c=cluster_labels, cmap='viridis')
            plt.title('KMeans Clustering (k=5)')
            plt.xlabel("Sun Azimuth")
            plt.ylabel("Sun Elevation")
            plt.savefig(r"/jump/tw/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/kmeans.png")
            plt.close()

            clustered_ids = {label: [] for label in range(n_clusters)}

            for cluster_label, calib_id in zip(cluster_labels, self.calibration_ids):
                clustered_ids[cluster_label].append(calib_id)

            train_ids = []
            valid_ids = []
            test_ids = []

            i = 0
            draw = True
            while draw:  # take turns drawing from clusters until all batches are full
                c = i % n_clusters
                cluster_label = cluster_labels[c]
                cluster_ids = clustered_ids[cluster_label]
                if len(train_ids) < train_size and cluster_ids:
                    train_ids.append(cluster_ids.pop())
                if len(valid_ids) < valid_size and cluster_ids:
                    valid_ids.append(cluster_ids.pop())
                if len(test_ids) < test_size and cluster_ids:
                    test_ids.append(cluster_ids.pop())
                if (len(train_ids) == train_size and len(valid_ids) == valid_size and len(test_ids) == test_size):
                    draw = False
                i += 1

            self.plot_splits_sun_positions(train_ids, valid_ids, test_ids)
            return train_ids, valid_ids, test_ids
    
    def plot_splits_sun_positions(self, train_ids: List[str], valid_ids: List[str], test_ids: List[str]) -> None:
        import matplotlib.pyplot as plt
        c_map = ['black', 'blue', 'red']
        labels = ['Training', 'Validation', 'Testing']
        for i, split in enumerate([train_ids, valid_ids, test_ids]):
            batch = self.get_batch(split)
            sun_azimuths_list = batch['sun_azimuths']
            sun_elevations_list = batch['sun_elevations']
            plt.scatter(sun_azimuths_list,
                    sun_elevations_list,
                    marker='o', color=c_map[i], alpha=0.8,
                    s=8, label=labels[i]
                    )
        plt.title(f"Sun Positions: Random Split")
        plt.xlabel("Sun Azimuth")
        plt.ylabel("Sun Elevation")
        plt.legend()
        plt.savefig(r"/jump/tw/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/sun_positions.png")
        plt.close()

    def __len__(self) -> int:
        """
        Get the number of calibration points.

        Returns:
            int: Number of calibration points
        """
        return len(self.calibration_ids)

    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Any]:
        """
        Get a single calibration data point.

        Args:
            idx (Union[int, str]): Either an integer index or a calibration ID string

        Returns:
            Dict: A dictionary containing the data for the requested calibration point

        Raises:
            KeyError: If the requested calibration ID is not available
            IndexError: If the index is out of range
        """
        if isinstance(idx, str):
            return self.get_single_item(idx)
        elif isinstance(idx, int):
            if idx < 0 or idx >= len(self.calibration_ids):
                raise IndexError(f"Index {idx} out of range for {len(self.calibration_ids)} items")
            return self.get_single_item(self.calibration_ids[idx])
        else:
            raise TypeError(f"Index must be an integer or string, got {type(idx)}")

