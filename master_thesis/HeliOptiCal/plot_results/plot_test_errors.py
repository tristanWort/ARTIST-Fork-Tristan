import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Union

import pandas as pd
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.colors import Normalize

from plot_results.plot_errors_distributions import plot_alignment_errors_over_sun_pos


def plot_alignment_error_comparison(test_errors: torch.Tensor,
                                     true_test_errors: torch.Tensor,
                                     heliostat_names: list,
                                     save_dir: Union[str, Path],
                                     log_scale: bool = False, blocking=None):
    """
    Generate comparison bar plots for test vs. true test alignment errors for each heliostat.

    Parameters
    ----------
    test_errors : torch.Tensor
        Tensor of shape [B, H] containing test alignment errors in mrad.
    true_test_errors : torch.Tensor
        Tensor of shape [B, H] containing true test alignment errors in mrad.
    heliostat_names : list
        List of heliostat names corresponding to axis H.
    save_dir : Union[str, Path]
        Directory to save the generated plots.
    """
    assert test_errors.shape == true_test_errors.shape, \
        "Predicted and true alignment errors must have the same shape [B, H]."
    assert test_errors.dim() == 2, "Input tensors must be of shape [B, H]."

    B, H = test_errors.shape
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    test_errors = test_errors.cpu().numpy()
    true_test_errors = true_test_errors.cpu().numpy()

    x = np.arange(B)

    for h in range(H):
        plt.figure(figsize=(10, 6))
        width = 0.4
        plt.bar(x - width / 2, test_errors[:, h], width, label='Error on Unit Centroid Vector', color='tab:blue')
        plt.bar(x + width / 2, true_test_errors[:, h], width, label='Error on Reflection Axis', color='tab:orange')
        
        if log_scale:
            plt.yscale('log')
            
        plt.title(f'Ratio of Artifical Blocking = {blocking} %')
        plt.xlabel('Index of Sample in Test Dataset')
        plt.ylabel('Final Alignment Error (mrad)')
        plt.xticks(x)
        plt.legend(loc='upper right')
        plt.tight_layout()

        filename = save_dir / f"{blocking}_alignment_errors_{heliostat_names[h]}.png"
        plt.savefig(filename, dpi=300)
        plt.close()


def plot_alignment_error_comparison_with_averages(test_errors: torch.Tensor,
                                     true_test_errors: torch.Tensor,
                                     heliostat_names: list,
                                     save_dir: Union[str, Path],
                                     log_scale: bool = False, blocking=None):
    """
    Generate comparison bar plots for test vs. true test alignment errors for each heliostat.

    Parameters
    ----------
    test_errors : torch.Tensor
        Tensor of shape [B, H] containing test alignment errors in mrad.
    true_test_errors : torch.Tensor
        Tensor of shape [B, H] containing true test alignment errors in mrad.
    heliostat_names : list
        List of heliostat names corresponding to axis H.
    save_dir : Union[str, Path]
        Directory to save the generated plots.
    """
    assert test_errors.shape == true_test_errors.shape, \
        "Predicted and true alignment errors must have the same shape [B, H]."
    assert test_errors.dim() == 2, "Input tensors must be of shape [B, H]."

    B, H = test_errors.shape
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    test_errors = test_errors.cpu().numpy()
    true_test_errors = true_test_errors.cpu().numpy()

    x = np.arange(B)

    for h in range(H):
        plt.figure(figsize=(10, 6))
        width = 0.4

        pred_color = 'tab:blue'
        true_color = 'tab:orange'

        plt.bar(x - width / 2, test_errors[:, h], width, label='Error on Unit Centroid Vector', color=pred_color)
        plt.bar(x + width / 2, true_test_errors[:, h], width, label='Error on Reflection Axis', color=true_color)

        # Calculate and plot average error lines
        avg_pred_error = test_errors[:, h].mean()
        avg_true_error = true_test_errors[:, h].mean()

        plt.axhline(avg_pred_error, color=pred_color, linestyle='dotted', linewidth=2,
                    label='Mean Error on Unit Centroid Vector')
        plt.axhline(avg_true_error, color=true_color, linestyle='dotted', linewidth=2,
                    label='Mean Error on Reflection Axis')

        if log_scale:
            plt.yscale('log')
            
        plt.title(f'Ratio of Artifical Blocking = {blocking} %')
        plt.xlabel('Index of Sample in Test Dataset')
        plt.ylabel('Alignment Error (mrad)')
        plt.xticks(x)
        plt.legend(loc='upper right')
        plt.tight_layout()

        filename = save_dir / f"{blocking}_alignment_errors_{heliostat_names[h]}_with_mean.png"
        plt.savefig(filename, dpi=300)
        plt.close()


def plot_comparative_sun_pos_error_distributions(test_errors: torch.Tensor,
                                                  true_test_errors: torch.Tensor,
                                                  data_batch_test: list,
                                                  heliostat_ids: list,
                                                  output_dir: Union[str, Path],
                                                  display_type: str = 'show_color_gradient'):
    """
    Visualize alignment error distributions over sun positions for each heliostat.

    Parameters
    ----------
    test_errors : torch.Tensor
        Predicted test alignment errors [B, H] in mrad.
    true_test_errors : torch.Tensor
        True test alignment errors [B, H] in mrad.
    data_batch_test : List[Dict]
        Batch dictionary from CalibrationModel with sun positions and calibration IDs.
    heliostat_ids : List[str]
        Ordered list of heliostat identifiers.
    output_dir : Union[str, Path]
        Directory where the plots will be saved.
    display_type : str
        Visualization mode: either 'show_size' or 'show_color_gradient'.
    """
    os.makedirs(output_dir, exist_ok=True)
    test_errors = test_errors.cpu().numpy()
    true_test_errors = true_test_errors.cpu().numpy()

    # Construct long-form DataFrame with one row per sample per heliostat
    merged_entries = []
    B, H = test_errors.shape
    for b in range(B):
        azimuths = data_batch_test[b]["sun_azimuths"]
        elevations = data_batch_test[b]["sun_elevations"]
        calib_ids = data_batch_test[b]["calibration_ids"]

        for h_idx, helio_id in enumerate(heliostat_ids):
            merged_entries.append({
                'sample': b,
                'heliostat_id': helio_id,
                'azimuth': azimuths[h_idx].cpu(),
                'elevation': elevations[h_idx].cpu(),
                'calib_id': calib_ids[h_idx],
                'test_error': test_errors[b, h_idx],
                'true_error': true_test_errors[b, h_idx],
                'mode': 'Test'  # Optional mode field for consistency
            })

    df_errors = pd.DataFrame(merged_entries)

    for error_type in ['test_error', 'true_error']:
        plot_alignment_errors_over_sun_pos(
            merged_data=df_errors.rename(columns={error_type: 'error'}),
            output_dir=os.path.join(output_dir, error_type),
            type=display_type
        )
        