import json
import os
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

from artist.util.paint_loader import  extract_paint_calibration_data
import paint.util.paint_mappings as mappings
from paint.data.dataset_splits import DatasetSplitter

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
    
    #TODO: Make CalibrationDataLoader a subclass of Dataset and implement __getitem__ and __len__ methods.
    def __init__(
            self,
            data_directory: Union[str, Path],
            heliostats_to_load: List[str],
            power_plant_position: torch.Tensor,
            load_flux_images: bool = True,
            ideal_flux_center: bool = False,
            properties_file_ends_with: Optional[str] = '-properties.json',
            flux_file_ends_with: Optional[str] = '-flux.png',
            device: Union[torch.device, str] = "cuda"
    ):
        """
        Initialize the CalibrationDataLoader with a data directory.

        Args:
            data_directory (str, Path): Path to the directory containing calibration data files with 'heliostat_id' as folders.
            heliostats_to_load (List[str]): List of heliostat IDs to load calibration data for.
            power_plant_position (torch.Tensor): The position of the power plant in the world coordinate system.
            load_flux_images (bool): Whether to load flux images from PNG files (default: True)
            properties_file_ends_with (str): File ending for calibration properties JSON files (default: '-calibration-properties.json')
            flux_file_ends_with (str): File ending for flux images PNG files (default: '-flux.png')
            device (torch.device, str): Device to load data tensors onto (default: 'cuda')
        """
        self.data_dir = data_directory
        self.heliostats_to_load = heliostats_to_load
        self.power_plant_position = power_plant_position

        # Storage dictionaries for each data type
        self.sun_azimuths: Dict[str, float] = {}
        self.sun_elevations: Dict[str, float] = {}
        self.flux_centers: Dict[str, torch.Tensor] = {}  # shape [4,]
        self.motor_positions: Dict[str, torch.Tensor] = {}  # shape [2,]
        self.incident_rays: Dict[str, torch.Tensor] = {}  # shape [4,]
        self.receiver_targets: Dict[str, str] = {}
        self.flux_images: Dict[str, torch.Tensor] = {}  # shape [256, 256]
        self.ideal_flux_centers: Dict[str, torch.Tensor] = {}
        
        # Whether to load flux images
        self.load_flux_images = load_flux_images
        
        # Whether properties files contain ideal flux centers
        self.has_ideal_flux_center = ideal_flux_center
        
        # File endings for calibration properties and flux images
        self.properties_file_ends_with = properties_file_ends_with
        self.flux_file_ends_with = flux_file_ends_with

        self.device = device

        # Load the data
        self._load_data()
        
        self.splits = {}

    def _load_data(self):
        """
        Load calibration data from JSON and PNG files in the data directory.

        This method should be customized to implement your specific data loading logic.
        """
        calibration_directories = {}
        for heliostat_id in self.heliostats_to_load: 
            calibration_directory = (Path(self.data_dir) /
                                    f'{heliostat_id}')
            if not os.path.exists(calibration_directory):
                raise FileNotFoundError(f"Calibration folder not found at path: "
                                        f"{calibration_directory}")
            else:
                calibration_directories[heliostat_id] = calibration_directory
        
        self.calibration_ids = {heliostat_id: [] for heliostat_id in self.heliostats_to_load}
        # Load all calibration json files for each heliostat.        
        for heliostat_id, calibration_directory in calibration_directories.items():
            log.info(f"Loading data for Heliostat {heliostat_id} from: {calibration_directory}")  
            properties_files = [os.path.join(calibration_directory, f)
                                for f in os.listdir(calibration_directory) 
                                if f.endswith(self.properties_file_ends_with)]

            
            # TODO: Change so that only data paths are stored and not the data itself.
            for properties_file in properties_files:
                calibration_id = int(''.join(filter(str.isdigit, Path(properties_file).name)))

                with open(properties_file, 'r') as file:
                    calibration_data = json.load(file)
                
                try:
                    if self.has_ideal_flux_center:
                        (
                            calibration_target_name,
                            spot_center,
                            sun_position,
                            motor_positions,
                            ideal_flux_center
                        ) = extract_paint_calibration_data(
                            calibration_properties_path=Path(properties_file),
                            power_plant_position=self.power_plant_position,
                            coord_system='local_enu',
                            has_ideal_flux_center=True,
                            device=self.device
                        )
                        self.ideal_flux_centers[calibration_id] = ideal_flux_center
                    else:
                        (
                            calibration_target_name,
                            spot_center,
                            sun_position,
                            motor_positions,
                        ) = extract_paint_calibration_data(
                            calibration_properties_path=Path(properties_file),
                            power_plant_position=self.power_plant_position,
                            coord_system='local_enu',
                            has_ideal_flux_center=False,
                            device=self.device
                        )
                except KeyError:
                    log.warning(f"Missing calibration data in {properties_file}. Skipping this file.")
                    continue
                
                self.calibration_ids[heliostat_id].append(calibration_id)
                self.sun_azimuths[calibration_id] = calibration_data['sun_azimuth']
                self.sun_elevations[calibration_id] = calibration_data['sun_elevation']
                self.flux_centers[calibration_id] = spot_center.to(self.device)
                self.motor_positions[calibration_id] = motor_positions.to(self.device)
                self.incident_rays[calibration_id] = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(self.device) - sun_position.to(self.device)
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
                                                f"{str(calibration_id) + self.flux_file_ends_with}.")
            log.info(f"Loaded data for {len(self.calibration_ids[heliostat_id])} calibration IDs.")

    def get_calibration_ids(self) -> Dict[str, int]:
        """
        Get all available calibration IDs.

        Returns:
            Dict[str, int]: Dict of all calibration IDs
        """
        return self.calibration_ids.copy()

    def get_batch(self, heliostat_id: str, cal_ids: List[int]) -> Dict[str, Union[torch.Tensor, List[str]]]:        
        
        # Check if all requested IDs are available
        missing_ids = set(cal_ids) - set(self.calibration_ids[heliostat_id])
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
                      'receiver_targets': batch_receiver_targets
                      }

        if self.load_flux_images:
            batch_flux_images = torch.stack([self.flux_images[cal_id] for cal_id in cal_ids]).to(self.device)
            batch_data['flux_images'] = batch_flux_images

        return batch_data
    
    def get_field_batch(self, heliostats_and_calib_ids: Dict[str, List[int]]) -> Dict[str, Dict[str, Union[torch.Tensor, List[str]]]]:

        # Check if all requested IDs are available
        missing_ids = (set(batch_id for ids in heliostats_and_calib_ids.values() for batch_id in ids)
                       - set(batch_id for ids in self.calibration_ids.values() for batch_id in ids))
        if missing_ids:
            raise KeyError(f"Calibration IDs not found in CalibrationDataLoader: {missing_ids}")

        sample_length = len(next(iter(heliostats_and_calib_ids.values())))
        
        field_batch = list()
        # Iterate over number of samples per heliostat
        for sample in range(sample_length):
            # Get the calibration id for one sample per heliostat
            sample_ids = [cal_ids[sample] for cal_ids in heliostats_and_calib_ids.values()]
            
            sample_flux_centers = torch.stack([self.flux_centers[id] for id in sample_ids]).to(self.device)
            sample_motor_positions = torch.stack([self.motor_positions[id] for id in sample_ids]).to(self.device)
            sample_incident_rays = torch.stack([self.incident_rays[id] for id in sample_ids]).to(self.device)
            sample_receiver_targets = [self.receiver_targets[id] for id in sample_ids]
            
            sample_data = {'cal_ids': sample_ids,
                          'sun_azimuths': [self.sun_azimuths[id] for id in sample_ids],
                          'sun_elevations': [self.sun_elevations[id] for id in sample_ids],
                          'flux_centers': sample_flux_centers,
                          'motor_positions': sample_motor_positions,
                          'incident_rays': sample_incident_rays,
                          'receiver_targets': sample_receiver_targets}
            
            if self.has_ideal_flux_center:
                ideal_flux_centers = torch.stack([self.ideal_flux_centers[id] for id in sample_ids]).to(self.device)
                sample_data['ideal_flux_centers'] = ideal_flux_centers

            if self.load_flux_images:
                sample_flux_images = torch.stack([self.flux_images[id] for id in sample_ids]).to(self.device)
                sample_data['flux_images'] = sample_flux_images
                
            field_batch.append(sample_data)

        return field_batch

    def get_single_item(self, cal_id: str) -> Dict[str, Any]:

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
        
        return [cal_id for cal_id in self.calibration_ids
                if self.receiver_targets[cal_id] == target_name]

    def random_batch(
            self,
            batch_size: int,
            seed: Optional[int] = None
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        
        if batch_size > len(self.calibration_ids):
            raise ValueError(
                f"Requested batch size {batch_size} exceeds available data size {len(self.calibration_ids)}")

        if seed is not None:
            torch.manual_seed(seed)

        # Sample random calibration IDs
        sample_ids = random.sample(range(len(self.calibration_ids)), batch_size)
        selected_ids = [self.calibration_ids[i] for i in sample_ids]

        return self.get_batch(selected_ids)

    def sun_positions_splits(
        self,
        config: Dict[str, Any],  # dictionary containing configuration for splits
        save_sun_positions_splits_plots = True,
    ):
        # Create a DatasetSplitter instance.
        # Use remove_unused_data=False to preserve extra columns (e.g. azimuth, elevation) needed for plotting.
        calibration_metadata_file = Path(config['path_to_measurements'])
        if not calibration_metadata_file.exists():
            raise FileNotFoundError(
                f"Calibration metadata file '{calibration_metadata_file}' not found."
            )

        output_dir = Path(config['output_dir'])
        
        splitter = DatasetSplitter(
            input_file=calibration_metadata_file,
            output_dir=output_dir,
            remove_unused_data=False,
        )

        # Read the full calibration metadata once.
        calibration_data = pd.read_csv(calibration_metadata_file)

        # Ensure that the plot_output directory exists.
        plot_output_path = Path(config['output_dir']) / "plots"
        plot_output_path.mkdir(parents=True, exist_ok=True)

        # For each split type, create a separate plot file.
        splits = {split_type: {} for split_type in config['split_types']}
        
        for split_type in config['split_types']:
            # For the current split type, gather the split data for each combination of training and validation sizes.
            # We use a dictionary keyed by (training_size, validation_size)
            current_split_data = {}
            for training_size in config['training_sizes']:
                for validation_size in config['validation_sizes']:
                    split_df = splitter.get_dataset_splits(
                        split_type=split_type,
                        training_size=training_size,
                        validation_size=validation_size,
                    )
                    
                    splits[split_type].update({(training_size, validation_size): split_df})
                    # split_df['SplitType'] = split_type
                    # split_df['Split'] = split_df['Split']
                    # split_df['Training'] = training_size
                    # split_df['Validation'] = validation_size
                    current_split_data[(training_size, validation_size)] = split_df
                    # splits_df = pd.concat([splits_df, split_df], ignore_index=True)
                
                # Append the IDs to the corresponding lists in the splits dictionary.
                # splits[split_type]['train'] = split_df[split_df['Split'] == 'train'].index.tolist()
                # splits[split_type]['validation'] = split_df[split_df['Split'] == 'validation'].index.tolist()
                # splits[split_type]['test'] = split_df[split_df['Split'] == 'test'].index.tolist()

            # Determine grid dimensions for subplots.
            # Here we use rows = number of validation sizes and columns = number of training sizes.
            if save_sun_positions_splits_plots == True:
                ncols = len(config['training_sizes'])
                nrows = len(config['validation_sizes'])
                num_plots = ncols * nrows

                fig, axes = plt.subplots(
                    nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows), sharey=True
                )

                # Flatten axes so that we can iterate uniformly.
                if num_plots == 1:
                    axes = [axes]
                else:
                    axes = np.array(axes).flatten()

                # For each combination, create a subplot.
                for ax, ((training_size, validation_size), split_df) in zip(
                    axes, current_split_data.items()
                ):
                    # Merge the split info into the full calibration data.
                    split_df_reset = (
                        split_df.reset_index()
                    )  # bring the ID (index) back as a column
                    merged_data = pd.merge(
                        calibration_data,
                        split_df_reset[[mappings.ID_INDEX, mappings.SPLIT_KEY]],
                        on=mappings.ID_INDEX,
                        how="left",
                    )

                    # Group by heliostat and split to count occurrences.
                    split_counts = (
                        merged_data.groupby([mappings.HELIOSTAT_ID, mappings.SPLIT_KEY])
                        .size()
                        .unstack(fill_value=0)
                    )
                    # Add total counts for sorting and then drop the helper column.
                    split_counts[mappings.TOTAL_INDEX] = split_counts.sum(axis=1)
                    split_counts = split_counts.sort_values(
                        by=mappings.TOTAL_INDEX, ascending=False
                    ).drop(columns=[mappings.TOTAL_INDEX])
                    # Reorder columns: train, then test, then validation.
                    split_counts = split_counts.reindex(
                        columns=[
                            mappings.TRAIN_INDEX,
                            mappings.TEST_INDEX,
                            mappings.VALIDATION_INDEX,
                        ],
                        fill_value=0,
                    )

                    # Replace the heliostat IDs with sequential numbers (for plotting purposes).
                    num_heliostats = len(split_counts)
                    split_counts.index = range(num_heliostats)

                    # Determine the bar colors using the shared mapping.
                    colors = mappings.TRAIN_TEST_VAL_COLORS
                    bar_colors = [colors.get(split, "gray") for split in split_counts.columns]

                    # Plot the stacked bar plot.
                    split_counts.plot(
                        kind="bar", stacked=True, ax=ax, legend=False, color=bar_colors
                    )
                    # Change the x-axis label as requested.
                    ax.set_xlabel("Heliostats sorted by # measurements", fontsize=10)
                    ax.set_ylabel("Count", fontsize=10)
                    ax.tick_params(axis="x", rotation=45)
                    ticks = list(range(0, num_heliostats, 200))
                    ax.set_xticks(ticks)

                    # Set y-axis limits for KNN and KMEANS split types.
                    if split_type in [mappings.KMEANS_SPLIT, mappings.KNN_SPLIT]:
                        ax.set_ylim(0, 500)

                    # Set subplot title indicating the training and validation sizes.
                    ax.set_title(f"Train {training_size} / Val {validation_size}", fontsize=12)

                    heliostat_ids = merged_data[mappings.HELIOSTAT_ID].unique()
                    # ---- Add an inset for the example heliostat ----
                    example_heliostat_df = merged_data[
                        merged_data[mappings.HELIOSTAT_ID] == heliostat_ids[0]
                    ]
                    inset_ax = inset_axes(
                        ax,
                        width="50%",
                        height="50%",
                        loc="upper right",
                        bbox_to_anchor=(0, -0.05, 1, 1),
                        bbox_transform=ax.transAxes,
                    )
                    for split, color in colors.items():
                        subset = example_heliostat_df[
                            example_heliostat_df[mappings.SPLIT_KEY] == split
                        ]
                        if not subset.empty:
                            inset_ax.scatter(
                                subset[mappings.AZIMUTH],
                                subset[mappings.ELEVATION],
                                color=color,
                                alpha=0.5,
                            )
                    inset_ax.set_title(f"Heliostat {heliostat_ids[0]}", fontsize=8, pad=-5)
                    inset_ax.set_xlabel("Azimuth", fontsize=8)
                    inset_ax.set_ylabel("Elevation", fontsize=8)
                    inset_ax.tick_params(axis="both", labelsize=8)

                # Create a common legend (placed in the upper left of the first subplot).
                legend_handles = [
                    mpatches.Patch(color=colors[split], label=split.capitalize())
                    for split in colors
                ]
                axes[0].legend(handles=legend_handles, loc="upper left", fontsize=10)

                plt.tight_layout()
                # Save the figure as "02_<split_type>_split.pdf"
                file_name = plot_output_path / f"02_{split_type}_split.pdf"
                plt.savefig(file_name, dpi=300)
                plt.close(fig)
                print(f"Saved plot for split type '{split_type}' to {file_name}")
        
        self.splits = splits  #TODO: Load the splits from dataframe or csv file.

    def split_data(
        self,
        train_valid_test_sizes: Tuple[int,int,int] = (30, 15, 15),
        split_type: Literal['random','kmeans','target'] = 'random',
        n_clusters: Optional[int] = 5,
        seed: Optional[int] = 42
    ) -> Tuple[List[str],List[str],Optional[List[str]]]:

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
    
    # TODO: Change plot function (Right now plotting for paint splits is dont in sun_positions_splits)
    def plot_splits_sun_positions(self, train_ids: List[str], valid_ids: List[str], test_ids: List[str]) -> None:
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
        #TODO: Change, ids is a dictionary.
        return len(self.calibration_ids)

    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Any]:

        if isinstance(idx, str):
            return self.get_single_item(idx)
        elif isinstance(idx, int):
            if idx < 0 or idx >= len(self.calibration_ids):
                raise IndexError(f"Index {idx} out of range for {len(self.calibration_ids)} items")
            return self.get_single_item(self.calibration_ids[idx])
        else:
            raise TypeError(f"Index must be an integer or string, got {type(idx)}")

