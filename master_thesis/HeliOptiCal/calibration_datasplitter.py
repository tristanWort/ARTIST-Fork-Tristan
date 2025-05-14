import json
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path
from typing import Union, Optional,Literal, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import paint.util.paint_mappings as mappings
from paint.data.dataset_splits import DatasetSplitter

import my_config_dict

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
# A logger for the calibration datasplitter.

class CalibrationDataSplitter:
    """
    Handles the splitting and visualization of calibration datasets based on sun positions 
    (azimuth and elevation). Wraps the DatasetSplitter logic and provides utilities for
    generating plots for each split.

    Parameters
    ----------
    metadata_path : Union[str, Path]
        Path to the calibration metadata CSV file.
    output_directory : Union[str, Path]
        Directory where split outputs and plots will be saved.
    """

    def __init__(self, metadata_path: Union[str, Path], output_directory: Union[str, Path]):
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Calibration metadata file '{self.metadata_path}' not found.")

        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Full calibration metadata for use in plotting
        self.calibration_data = pd.read_csv(self.metadata_path)

        # DatasetSplitter instance
        self.splitter = DatasetSplitter(
            input_file=self.metadata_path,
            output_dir=self.output_directory,
            remove_unused_data=False,
        )
        
        # Save splits
        self.splits = dict()

    def plot_individual_heliostat_scatter(self,
                                           split_type: str,
                                           heliostat_id: str,
                                           merged_data: pd.DataFrame,
                                           training_size: int,
                                           validation_size: int):
        """
        Generate and save a scatter plot of sun positions for one heliostat.
        """
        colors = mappings.TRAIN_TEST_VAL_COLORS
        heliostat_df = merged_data[merged_data[mappings.HELIOSTAT_ID] == heliostat_id]

        plt.figure(figsize=(6, 5))
        for split, color in colors.items():
            subset = heliostat_df[heliostat_df[mappings.SPLIT_KEY] == split]
            if not subset.empty:
                plt.scatter(subset[mappings.AZIMUTH], subset[mappings.ELEVATION],
                            color=color, alpha=0.5, label=split)

        plt.title(f"Heliostat {heliostat_id}\nTrain {training_size} / Val {validation_size}")
        plt.xlabel("Azimuth")
        plt.ylabel("Elevation")
        plt.legend()
        plt.tight_layout()

        file_path = self.output_directory / f"heliostat_{heliostat_id}_split_{split_type}_train{training_size}_val{validation_size}.png"
        plt.savefig(file_path, dpi=200)
        plt.close()
        log.info(f"Saved scatter plot for Heliostat {heliostat_id} as {file_path.name}.")

    def plot_sun_positions_splits(self,
                                   split_type: str,
                                   split_data: Dict,
                                   training_sizes: List[int],
                                   validation_sizes: List[int]):
        """
        Generate bar plots and heliostat scatter plots for sun position splits.
        """
        ncols = len(training_sizes)
        nrows = len(validation_sizes)
        num_plots = ncols * nrows

        fig, axes = plt.subplots(nrows=nrows, 
                                 ncols=ncols, 
                                 figsize=(6 * ncols, 5 * nrows), 
                                 sharey=True)

        if num_plots == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()

        for ax, ((training_size, validation_size), split_df) in zip(axes, split_data.items()):
            split_df_reset = split_df.reset_index()
            merged_data = pd.merge(
                self.calibration_data,
                split_df_reset[[mappings.ID_INDEX, mappings.SPLIT_KEY]],
                on=mappings.ID_INDEX,
                how="left"
            )

            split_counts = (
                merged_data.groupby([mappings.HELIOSTAT_ID, mappings.SPLIT_KEY])
                .size()
                .unstack(fill_value=0)
            )
            split_counts[mappings.TOTAL_INDEX] = split_counts.sum(axis=1)
            split_counts = split_counts.sort_values(by=mappings.TOTAL_INDEX, ascending=False).drop(columns=[mappings.TOTAL_INDEX])
            split_counts = split_counts.reindex(columns=[
                mappings.TRAIN_INDEX, mappings.TEST_INDEX, mappings.VALIDATION_INDEX
            ], fill_value=0)

            num_heliostats = len(split_counts)
            split_counts.index = range(num_heliostats)

            bar_colors = [mappings.TRAIN_TEST_VAL_COLORS.get(split, "gray") for split in split_counts.columns]

            split_counts.plot(kind="bar", stacked=True, ax=ax, legend=False, color=bar_colors)
            ax.set_xlabel("Heliostats sorted by # measurements", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.tick_params(axis="x", rotation=45)
            ax.set_xticks(list(range(0, num_heliostats, 200)))

            if split_type in [mappings.KMEANS_SPLIT, mappings.KNN_SPLIT]:
                ax.set_ylim(0, 300)

            ax.set_title(f"Train {training_size} / Val {validation_size}", fontsize=12)

            heliostat_ids = merged_data[mappings.HELIOSTAT_ID].unique()
            for heliostat_id in heliostat_ids:
                if not pd.isna(heliostat_id):
                    self.plot_individual_heliostat_scatter(
                        split_type=split_type,
                        heliostat_id=heliostat_id,
                        merged_data=merged_data,
                        training_size=training_size,
                        validation_size=validation_size
                    )

        legend_handles = [
            mpatches.Patch(color=mappings.TRAIN_TEST_VAL_COLORS[split], label=split.capitalize())
            for split in mappings.TRAIN_TEST_VAL_COLORS
        ]
        axes[0].legend(handles=legend_handles, loc="upper left", fontsize=10)

        plt.tight_layout()
        file_path = self.output_directory / f"{split_type}_all_splits.png"
        plt.savefig(file_path, dpi=300)
        plt.close(fig)
        log.info(f"Saved bar plots for split type '{split_type}' as {file_path.name}")

    def perform_splits(self,
                       training_sizes: List[int] = (15, 30, 50),
                       validation_sizes: List[int] = (30),
                       split_types: List[str] = ('azimuth', 'solstice', 'kmeans', 'knn'),
                       save_splits_plots: bool = True):
        """
        Perform sun-position-based splits across multiple strategies and optionally generate visualizations.

        Returns
        -------
        Dict[str, Dict[Tuple[int, int], pd.DataFrame]]
            Nested dictionary mapping split type -> (train, val size) -> split DataFrame
        """
        splits = {split_type: {} for split_type in split_types}

        for split_type in split_types:
            current_split_data = {}
            for training_size in training_sizes:
                for validation_size in validation_sizes:
                    if split_type == my_config_dict.RANDOM_SPLIT:
                        split_df = self.get_random_split(
                            training_size=training_size,
                            validation_size=validation_size
                        )
                    else:
                        split_df = self.splitter.get_dataset_splits(
                            split_type=split_type,
                            training_size=training_size,
                            validation_size=validation_size
                        )
                    current_split_data[(training_size, validation_size)] = split_df
                    splits[split_type][(training_size, validation_size)] = split_df

            if save_splits_plots:
                self.plot_sun_positions_splits(
                    split_type=split_type,
                    split_data=current_split_data,
                    training_sizes=training_sizes,
                    validation_sizes=validation_sizes
                )

        self.splits = splits
    
    def get_random_split(self, training_size: int, validation_size: int):
        """
        Split dataset randomly and return Dataframe with train, validation and test splits.
        Each Heliostat's calibration data is randomly sampled for training size and validation size. The remaining data
        will be the test split.
        """
        log.info("Preparing random split... \n" \
                 f"Training size: {training_size}, validation size: {validation_size}, and test size > {validation_size}")
        # Ensure 'Id' is the index
        calibration_data = self.calibration_data.set_index(mappings.ID_INDEX, drop=True)
        heliostat_ids = calibration_data[mappings.HELIOSTAT_ID].unique()

        split_dataframes = []
        min_images_required = training_size + validation_size

        for helio_id in heliostat_ids:
            helio_data = calibration_data[calibration_data[mappings.HELIOSTAT_ID] == helio_id].copy()
            
            if len(helio_data) < min_images_required:
                log.info(f"Heliostat {helio_id} has less than the required calibration data and will be skipped.")
                continue  # skip if not enough samples

            sampled = helio_data.sample(
                n=len(helio_data),
                random_state=42  # for reproducibility
            )
            indices = sampled.index.tolist()

            train_idx = indices[:training_size]
            val_idx = indices[training_size:training_size + validation_size]
            test_idx = indices[training_size + validation_size:]

            helio_data.loc[train_idx, mappings.SPLIT_KEY] = mappings.TRAIN_INDEX
            helio_data.loc[val_idx, mappings.SPLIT_KEY] = mappings.VALIDATION_INDEX
            helio_data.loc[test_idx, mappings.SPLIT_KEY] = mappings.TEST_INDEX

            split_dataframes.append(helio_data.loc[train_idx + val_idx + test_idx])

        if not split_dataframes:
            raise ValueError("No valid heliostats found for random split. Reduce split sizes.")

        return pd.concat(split_dataframes).sort_index()
    
    def get_helio_and_calib_ids_from_split(self, 
                                           split_type: Literal['azimuth', 'solstice', 'kmeans', 'knn'], 
                                           split_size: Tuple[int],
                                           split: Literal['train', 'validaton', 'test'] = None,
                                           heliostat_ids: Optional[List[str]] = None):
        """
        Retrieve a specific split and return the Heliostat and calibration IDs.
        
        Parameters
        ----------
        split_type : Literal['azimuth', 'solstice', 'kmeans', 'knn']
            The strategy used to generate the data split.
        split_size : Tuple[int]
            The sample size given as a tuple of training size and validation size.
        split : Literal['train', 'validation', 'test']
            The split subset to retrieve.
        heliostat_ids : Optional[List[str]]
            If given then only data for these heliostats will be loaded (default is None).
            
        Returns
        -------
        helio_and_calib_ids : Dict[str, List[int]]
            A dictionary of Heliostat IDs as keys and respective samples of calibration IDs as values.
            Pass to method `get_field_batch` of class-instance `CalibrationDataLoader` to receive data batch.
        """
        if not hasattr(self, 'splits'):
            raise AttributeError("Splits not found. Run `perform_splits()` first.")

        if split_type not in self.splits:
            raise ValueError(f"Split type '{split_type}' not found. Available types: {list(self.splits.keys())}")

        if split_size not in self.splits[split_type]:
            raise ValueError(f"No splits found for split size {split_size}.")
        
        split_df = self.splits[split_type][split_size]
        if split is not None:
            split_df = split_df[split_df['Split'] == split]
        
        # If no Heliostat IDs were specified, load data for all Heliostats. 
        if heliostat_ids is None:
            heliostat_ids = [id for id in split_df[mappings.HELIOSTAT_ID].unique()
                             if not pd.isna(id)]
        
        # Generate a dictionary of Heliostat IDs and respective samples for calibration IDs
        # which can be used in the `CalibrationDataLoader` to load data batch.
        helio_and_calib_ids = {
            heliostat_id: split_df.loc[split_df[mappings.HELIOSTAT_ID] == heliostat_id].index.tolist()
            for heliostat_id in heliostat_ids
        }
        return helio_and_calib_ids

        