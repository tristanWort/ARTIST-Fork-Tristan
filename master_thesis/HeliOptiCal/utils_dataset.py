import json
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path
from typing import Union, Literal, Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from artist.util import config_dictionary, utils
import paint.util.paint_mappings as mappings
from paint.data.dataset_splits import DatasetSplitter

import config_extended

log = logging.getLogger(__name__)

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
            calibration_dict[config_extended.focal_spot_enu_4d],
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


def plot_individual_heliostat_scatter(split_type: str, 
                                      heliostat_id: str,
                                      merged_data: pd.DataFrame,  
                                      training_size: int, 
                                      validation_size: int, 
                                      output_directory: Union[Path, str]):
    """
    Generate a scatter plot for the sun distribution of one split and one Heliostat.
    Plot will be saved to the output directory.
    """
    colors = mappings.TRAIN_TEST_VAL_COLORS
    heliostat_df = merged_data[merged_data[mappings.HELIOSTAT_ID] == heliostat_id]

    plt.figure(figsize=(6, 5))
    for split, color in colors.items():
        subset = heliostat_df[heliostat_df[mappings.SPLIT_KEY] == split]
        if not subset.empty:
            plt.scatter(subset[mappings.AZIMUTH], subset[mappings.ELEVATION], color=color, alpha=0.5, label=split)

    plt.title(f"Heliostat {heliostat_id}\nTrain {training_size} / Val {validation_size}")
    plt.xlabel("Azimuth")
    plt.ylabel("Elevation")
    plt.legend()
    plt.tight_layout()

    filename = output_directory / f"heliostat_{heliostat_id}_split_{split_type}_train{training_size}_val{validation_size}.png"
    plt.savefig(filename, dpi=200)
    plt.close()
    log.info(f"Saved scatter plot for Heliostat {heliostat_id} to {filename}")


def plot_sun_positions_splits(split_type: str,
                              split_data: Dict,
                              calibration_metadata: pd.DataFrame,
                              training_sizes: List,
                              validation_sizes: List,
                              output_directory: Union[Path, str]):
    """
    Generate bar plot for one split type showing all splits. Also, generate plot for sun distributions for each split and each heliostat.
    Save these plots to the selected directory as PNGs. 
    """
    # Determine grid dimensions for subplots.
    # Here we use rows = number of validation sizes and columns = number of training sizes. 
    ncols = len(training_sizes)
    nrows = len(validation_sizes)
    num_plots = ncols * nrows

    fig, axes = plt.subplots(nrows=nrows, 
                             ncols=ncols, 
                             figsize=(6 * ncols, 5 * nrows), 
                             sharey=True)

    # Flatten axes so that we can iterate uniformly.
    if num_plots == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    # For each combination, create a subplot.
    for ax, ((training_size, validation_size), split_df) in zip(
        axes, split_data.items()
        ):
        # Merge the split info into the full calibration data.
        split_df_reset = (
            split_df.reset_index()
            )  # bring the ID (index) back as a column
        merged_data = pd.merge(
            calibration_metadata,
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
            ax.set_ylim(0, 300)

        # Set subplot title indicating the training and validation sizes.
        ax.set_title(f"Train {training_size} / Val {validation_size}", fontsize=12)

        heliostat_ids = merged_data[mappings.HELIOSTAT_ID].unique()
        for heliostat_id in heliostat_ids:
            if not pd.isna(heliostat_id):
                plot_individual_heliostat_scatter(split_type=split_type,
                                                heliostat_id=heliostat_id,
                                                merged_data=merged_data, 
                                                training_size=training_size, 
                                                validation_size=validation_size, 
                                                output_directory=output_directory)

    # Create a common legend (placed in the upper left of the first subplot).
    legend_handles = [
        mpatches.Patch(color=colors[split], label=split.capitalize())
        for split in colors
        ]
    axes[0].legend(handles=legend_handles, loc="upper left", fontsize=10)

    plt.tight_layout()
    # Save the figure
    file_name = output_directory / f"{split_type}_all_splits.png"
    plt.savefig(file_name, dpi=300)
    plt.close(fig)
    print(f"Saved bar plots for split type '{split_type}' to {file_name}")


def sun_positions_splits(path_to_metadata: Union[Path, str],
                         output_directory: Union[Path, str],
                         training_sizes = [15, 30, 50],
                         validation_sizes = [30],
                         split_types = ['azimuth', 'solstice', 'kmeans', 'knn'],
                         save_splits_plots = True):
    """
    Copied function from ``PAINT`` for splitting calibration data along sun positions.
    """
    calibration_metadata_file = Path(path_to_metadata)
    if not calibration_metadata_file.exists():
        raise FileNotFoundError(
            f"Calibration metadata file '{calibration_metadata_file}' not found."
        )

    # Set output directory for saving the split plots
    output_directory = Path(output_directory)
    
    # Create a DatasetSplitter instance.
    # Use remove_unused_data=False to preserve extra columns (e.g. azimuth, elevation) needed for plotting.
    splitter = DatasetSplitter(
        input_file=calibration_metadata_file,
        output_dir=output_directory,
        remove_unused_data=False,
    )

    # Read the full calibration metadata once.
    calibration_data = pd.read_csv(calibration_metadata_file)

    # Ensure that the plot_output directory exists.
    output_directory.mkdir(parents=True, exist_ok=True)

    # For each split type, create a separate plot file.
    splits = {split_type: {} for split_type in split_types}
    
    for split_type in split_types:
        # For the current split type, gather the split data for each combination of training and validation sizes.
        # We use a dictionary keyed by (training_size, validation_size)
        current_split_data = {}
        for training_size in training_sizes:
            for validation_size in validation_sizes:
                split_df = splitter.get_dataset_splits(
                    split_type=split_type,
                    training_size=training_size,
                    validation_size=validation_size,
                )
                
                splits[split_type].update({(training_size, validation_size): split_df})
                current_split_data[(training_size, validation_size)] = split_df

        # Create plots to current split if these should be saved
        if save_splits_plots == True:
            plot_sun_positions_splits(split_type=split_type,
                                      split_data=current_split_data, 
                                      calibration_metadata=calibration_data,
                                      training_sizes=training_sizes,
                                      validation_sizes=validation_sizes,
                                      output_directory=output_directory)
    return splits
                
