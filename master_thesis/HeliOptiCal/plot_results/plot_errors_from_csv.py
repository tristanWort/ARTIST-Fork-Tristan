import pandas as pd 
import torch
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import Optional, List, Tuple

# Add local artist path for raytracing with multiple parallel heliostats.
artist_repo = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, artist_repo)  
from artist.util import utils

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, repo_path)
from HeliOptiCal.plot_results.plot_errors_distributions import *


def load_avg_alignment_errors(csv_path: str, correct_eps_clamp=False):
    """
    Load average alignment errors from a CSV file. The first row contains Heliostat IDs,
    the second row indicates the dataset mode ('Train', 'Valid', 'Test').

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: ['heliostat_id', 'mode', 'error']
    """
    df_raw = pd.read_csv(csv_path, header=None)
    heliostat_ids = df_raw.iloc[0, 1:].tolist()  # skip first column
    modes = df_raw.iloc[1, 1:].tolist()
    errors = df_raw.iloc[-1, 1:].astype(float).tolist()
    
    if correct_eps_clamp:
        eps_clamp = 1e-8
        correction_mrad = np.arccos(1 - eps_clamp) * 1000  # radians to milliradians
        errors = [max(e - correction_mrad, 0.0) for e in errors]
        
    data = {
        "heliostat_id": heliostat_ids,
        "mode": modes,
        "error": errors
    }

    df = pd.DataFrame(data)
    return df


def load_final_avg_alignment_errors(csv_path: str, mode='test'):
    """
    Load average alignment errors from a CSV file. The first row contains Heliostat IDs,
    the second row indicates the dataset mode ('Train', 'Valid', 'Test').

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: ['heliostat_id', 'mode', 'error']
    """
    df = load_avg_alignment_errors(csv_path)
    df_test = df[df["mode"].str.lower() == mode.lower()].reset_index(drop=True)
    return df_test


def load_alignment_errors(csv_path: str, idx=-1):
    """
    Load  alignment errors from a CSV file. The first row contains Heliostat IDs,
    the second row indicates the dataset mode ('Train', 'Valid', 'Test'),
    and third row indicates the calibration ID.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: ['heliostat_id', 'mode', 'id', 'error']
    """
    df_raw = pd.read_csv(csv_path, header=None)
    heliostat_ids = df_raw.iloc[0, 1:].tolist()  # skip first column
    modes = df_raw.iloc[1, 1:].tolist()
    calib_ids = df_raw.iloc[2, 1:].tolist()
    errors = df_raw.iloc[idx, 1:].astype(float).tolist()

    data = {
        "heliostat_id": heliostat_ids,
        "mode": modes,
        "calib_id": calib_ids,
        "error": errors
    }

    df = pd.DataFrame(data)
    return df


def load_tower_measurements(tower_masurements_path: str):
    """
    Return a dictionary of Solar Towers with coordinates.
    """
    device = torch.device('cpu')
    tower_dict = json.load(open(tower_masurements_path, 'r')) 
    power_plant_position = power_plant_position = torch.tensor(tower_dict['power_plant_properties']['coordinates'], 
                                                               dtype=torch.float64, 
                                                               device=device)
    # Load Target Tower data
    receiver_towers = []
    solar_tower_juelich_position =  torch.tensor(tower_dict['solar_tower_juelich_upper']['coordinates']['center'], 
                                                 dtype=torch.float64, 
                                                 device=device)
    mutlifocus_tower_position = torch.tensor(tower_dict['multi_focus_tower']['coordinates']['center'], 
                                             dtype=torch.float64, 
                                             device=device)
    receiver_towers.append({'id': 'Solar Tower Juelich', 'wgs84': solar_tower_juelich_position, 'east': 0, 'north': 0})
    receiver_towers.append({'id': 'Multifocus Tower', 'wgs84': mutlifocus_tower_position, 'east': 0, 'north': 0})
    # Tower coordinates
    for tower in receiver_towers:
        tower_wgs84_pos = tower['wgs84']
        tower_pos_enu = utils.convert_wgs84_coordinates_to_local_enu(tower_wgs84_pos, power_plant_position, device=device)
        tower['east'] = tower_pos_enu[0].item()
        tower['north'] = tower_pos_enu[1].item()    
    return receiver_towers


def load_heliostat_positions(metadata_path: str, tower_masurements_path: str):
    device = torch.device('cpu')
    # Load the Heliostat deflectometry data
    df = pd.read_csv(metadata_path)
    df = df[['HeliostatId', 'latitude', 'longitude']].dropna()
    df = df.drop_duplicates(subset='HeliostatId', keep='first').reset_index(drop=True)
    df = df.rename(columns={'HeliostatId': 'heliostat_id'})  # Rename column

    # Load power plant position
    tower_dict = json.load(open(tower_masurements_path, 'r')) 
    power_plant_position = power_plant_position = torch.tensor(tower_dict['power_plant_properties']['coordinates'], 
                                                               dtype=torch.float64, 
                                                               device=device)    
    # Heliostat coordinates
    east_list = []
    north_list = []
    for _, row in df.iterrows():
        heliostat_pos_wgs = torch.tensor([row['latitude'], row['longitude'], 0.0], dtype=torch.float64, device=device)
        heliostat_pos_enu = utils.convert_wgs84_coordinates_to_local_enu(heliostat_pos_wgs, power_plant_position, device=device)
        east_list.append(heliostat_pos_enu[0].item())
        north_list.append(heliostat_pos_enu[1].item())
    df['east'] = east_list
    df['north'] = north_list
    return df


def plot_splits(alignment_errors: pd.DataFrame, output_dir: str, all_sun_positions_df: Optional[pd.DataFrame] = None):
    """
    Plot the sun positions splits including the excluded samples if all_sun_positions_df is not None.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique heliostat IDs
    heliostat_ids = alignment_errors['heliostat_id'].unique()
    
    cmap = {'Train': 'blue', 'Test': 'red', 'Valid': 'green', 'Excluded': 'grey'}
    
    for heliostat_id in heliostat_ids:
        calib_ids = alignment_errors['calib_id'].values
        # Filter data for this heliostat
        heliostat_data = all_sun_positions_df[all_sun_positions_df['HeliostatId'] == heliostat_id]
              
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Iterate over modes ['Train', 'Valid', 'Test']
        for _, data in heliostat_data.iterrows():
            
            calib_id = data['calib_id']
            if str(calib_id) in calib_ids: 
                heliostat_match =  alignment_errors[alignment_errors['heliostat_id'] == heliostat_id]
                found_row = heliostat_match[heliostat_match['calib_id'] == str(calib_id)]
                mode = found_row['mode'].iloc[0]
            else:
                mode = "Excluded"

            ax.plot(data['azimuth'], data['elevation'], color=cmap[mode], marker='o', alpha=0.5)

        map_mode = {'Train': 'Training', 'Valid': 'Validation', 'Test': 'Testing', 'Excluded': 'Excluded'}
        for mode, color in cmap.items():
            ax.scatter([], [], color=color, alpha=0.5, label=map_mode[mode])

        
        # Add legend for modes
        ax.legend(fontsize=14)
        ax.set_title(f'K-NN Split on Euler Angles for Heliostat {heliostat_id}', fontsize=16)
        ax.set_xlabel('Azimuth [°]', fontsize=14)
        ax.set_ylabel('Elevation [°]', fontsize=14)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{heliostat_id}_split_sun_distributions.png'), dpi=300)
        plt.close(fig)
        
        print(f"Plot for heliostat {heliostat_id} saved to {output_dir}")


def plot_test_errors_comparison_over_field(df_actual_errors: pd.DataFrame, df_positions: pd.DataFrame,
                                           title: str, output_dir: str, save_under: str, marker_scaling: float = 10.0,
                                           df_errors: Optional[pd.DataFrame] = None, tower_positions: Optional[list] = None):
    """
    Plot the final average test alignment errors over the ENU field layout of heliostats.

    Parameters
    ----------
    df_actual_errors : pd.DataFrame
        DataFrame with columns 'heliostat_id', 'error' (for mode 'Test') for errors on reflection axis (ie. 'actual' errors).
    df_positions : pd.DataFrame
        DataFrame with columns 'heliostat_id', 'east', 'north' for Heliostat positions in field.
    output_path : str
        File path to save the output plot.
    scaling_factor : float
        Scaling for circle size per mrad error (default is 20.0)
    df_errors : pd.DataFrame (Optional)
        DataFrame with columns 'heliostat_id', 'error' (for mode 'Test') for measured errors on unit centroid axis.
    tower_positions : list (Optional)
        List of dictionaries containing solar tower position with 'id', 'east', 'north'. 
    """
    df = pd.merge(df_actual_errors, df_positions, on="heliostat_id", how="inner")

    max_error = df["error"].max()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for _, row in df.iterrows():
        east = row["east"]
        north = row["north"]
        error = row["error"]
        heliostat_id = row["heliostat_id"]

        # Draw concentric grey circles at multiples of 1 mrad
        ring_radius = 1.0
        while ring_radius <= (error + 1.0):
            ax.plot(east, north, color='grey', fillstyle='none', marker='o', markersize=ring_radius * marker_scaling, alpha=0.4)
            ring_radius += 1.0

        # Plot filled circle with error-based size for actual errors
        ax.plot(east, north, color='red', marker='o', markersize=error * marker_scaling, alpha=0.6)
        
        # Plot measured error if given
        if df_errors is not None:
            measured_err = df_errors[df_errors["heliostat_id"]==heliostat_id]["error"]
            ax.plot(east, north, color='blue', fillstyle='none', marker='o', markersize=measured_err * marker_scaling, alpha=0.8)

        # Annotate with heliostat_id
        ax.text(east + 3, north  + 3, heliostat_id, fontsize=11)
    
    # Include tower positions if given
    tower_colors = ['olive', 'darkorange']
    if tower_positions is not None:
        for idx, tower in enumerate(tower_positions):
            ax.scatter(tower['east'], tower['north'], marker='P', s=200, color=tower_colors[idx % len(tower_colors)], label=tower['id'])
            
    ax.set_xlabel("East [m]", fontsize=11)
    ax.set_ylabel("North [m]", fontsize=11)
    ax.set_title(f"{title}", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Add a legend marker for scale
    ax.scatter([], [], s=1.0 * marker_scaling, facecolors='none', edgecolors='red', label='Error on Reflection Axis')
    ax.scatter([], [], s=1.0 * marker_scaling, facecolors='none', edgecolors='blue', label="Error on UNet-Centroid Axis")
    ax.legend(loc='lower right', fontsize=11)
    
    # Show one empty circle for scaling in upper right corner
    east_min, east_max = df["east"].min(), df["east"].max()
    north_min, north_max = df["north"].min(), df["north"].max()
    ax.plot(120, 30, color='grey', fillstyle='none', marker='o', markersize=1 * marker_scaling, alpha=0.4)
    ax.text(125, 30 -2, '= 1 mrad', fontsize=11)
    
    # Scale axis
    ax.set_xlim(east_min - 25, east_max + 25)
    ax.set_ylim(- 20, north_max + 25)
    
    output_path = output_dir + f'/{save_under}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved test error plot to: {output_path}")


def plot_error_histograms_per_heliostat(df_errors: pd.DataFrame, output_dir: str, bin_width: float = 0.01):
    """
    Create and save a histogram of alignment errors for each heliostat individually.
    """
    # os.makedirs(output_dir, exist_ok=True)
    heliostat_ids = df_errors['heliostat_id'].unique()

    for heliostat_id in heliostat_ids:
        errors = df_errors[df_errors['heliostat_id'] == heliostat_id]['error'].astype(float)

        plt.figure(figsize=(8, 5))
        bins = int((errors.max() - errors.min()) / bin_width) + 1
        plt.hist(errors, bins=bins, range=(0, errors.max()), edgecolor='black', alpha=0.75)
        plt.title(f'Alignment Error Distribution – Heliostat {heliostat_id}')
        plt.xlabel('Alignment Error [mrad]')
        plt.ylabel('Absolute Frequency')
        plt.grid(True, linestyle='--', alpha=0.5)

        save_path = os.path.join(output_dir, f'error_histogram_{heliostat_id}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved histogram for Heliostat {heliostat_id} at {save_path}")


def plot_combined_error_histogram(df_errors: pd.DataFrame, output_path: str, bin_width: float = 0.1):
    """
    Create and save a single histogram of all alignment errors across all heliostats.
    """
    errors = df_errors['error'].astype(float)

    plt.figure(figsize=(10, 6))
    bins = int((errors.max() - errors.min()) / bin_width) + 1
    plt.hist(errors, bins=bins, range=(0, errors.max()), edgecolor='blue', alpha=0.5, color='steelblue')
    plt.title('Distribution of Tracking Errors in Final Testing across Heliostat Field')
    plt.xlabel('Tracking Error [mrad]')
    plt.ylabel('Absolute Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    # plt.savefig(output_path, dpi=300)

    plt.savefig(output_path, format="pdf")
    plt.close()
    print(f"Saved combined histogram to: {output_path}")


def plot_multiple_error_distributions(df_list: List[pd.DataFrame], labels: List[str], output_path: str, scenario: str, bin_width: float = 0.1, 
                                      plot_title: bool = True,
                                      df_list_2: Optional[List[pd.DataFrame]] = None):
    """
    Plot multiple alignment error distributions on a shared histogram.

    Parameters
    ----------
    df_list : List[pd.DataFrame]
        List of dataframes, each containing a column 'error' with alignment errors (in mrad).
    labels : List[str]
        List of names for each distribution, shown in the legend.
    output_path : str
        Path to save the resulting plot (PDF).
    senario : str
        Scenario Name, e.g. 'Base'
    bin_width : float
        Width of histogram bins in mrad.
    df_list_2 : Optional[List[pd.DataFrame]]
        Optional second list of dataframes. If given, the plot will show the histograms on a 2x1 grid.
    """
    assert len(df_list) == len(labels), "Number of dataframes must match number of labels."
    if df_list_2 is not None:
        assert len(df_list_2) == len(labels), "Number of dataframes must match number of labels."
    
    if df_list_2 is None:
        # Combine to determine consistent bin range
        all_errors = pd.concat([df['error'].astype(float) for df in df_list])
        min_val, max_val = all_errors.min(), all_errors.max()
        bins = int((max_val - min_val) / bin_width) + 1

        plt.figure(figsize=(10, 6))

        # Select cmap
        cmap = cm.get_cmap('tab10')  # 'tab10' is good for categorical colors
        colors = [mcolors.to_hex(cmap(i % 10)) for i in range(len(df_list))]
        colors.insert(0, colors.pop())
        
        plt.scatter([], [], label=r'$\text{Alignment-Loss}(\vec{r}_\text{G}, \vec{r}_\text{P})=\text{MSE}(1, \text{COS-SIM}(\vec{r}_\text{G}, \vec{r}_\text{P}))$', 
                       alpha=0.0)
        
        for idx, df in enumerate(df_list):
            trained_on_ideal = True if idx == 0 else False  
            errors = df['error'].astype(float)
            label = labels[idx]
            color = colors[idx]
            
            mean_alpha = 0.6
            mean_linewidth = 3.5
            mean_val = errors.mean()
            label_mean = f"Mean: {mean_val:.2f} mrad"
            
            if not trained_on_ideal:
                mean_linestyle = ['--', '-.', ':', '--', '-.', ':'][idx]
                plt.hist(errors, bins=bins, range=(0, max_val), alpha=0.5, 
                         edgecolor='black', label=label, color=color)
            else:
                mean_linestyle = '-'
                mean_alpha = 1.0
                mean_linewidth = 1.5
                if 'base' in scenario.lower():
                    n, bins, patches = plt.hist(errors, bins=bins, range=(0, max_val), alpha=0.5, 
                                edgecolor='black', label=label, color=color)
                    for patch in patches:
                        patch.set_hatch('//')
                else:
                    label_mean = label + f" Mean: {mean_val:.2f} mrad"
                    
            plt.axvline(mean_val, color=color, linestyle=mean_linestyle, linewidth=mean_linewidth, alpha=mean_alpha, label=label_mean)
            
        plt.ylabel('Absolute Frequency', fontsize=11)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlabel(r'$\text{Tracking Error [mrad] := }\theta(\vec{r}_\text{G}, \vec{r}_\text{P})$', fontsize=11)
        if plot_title:
            plt.title(f"Tracking Errors over Testing Samples for '{scenario}'-Scenario", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, format="pdf")
        plt.close()
        print(f"Saved multiple-distribution histogram to: {output_path}")
        
    else:
        # Two subplots [2, 1]
        rows = 2 if df_list_2 is not None else 1
        size = 10 if df_list_2 is not None else 6
        fig, axs = plt.subplots(rows, 1, figsize=(10, size), sharex=False)
        cmap = cm.get_cmap('tab10')

        x_labels = [r'$\text{Tracking Error [mrad] := }\theta(\vec{r}_\text{G}, \vec{r}_\text{P})$']
        # First subplot
        if df_list_2 is not None:
            x_labels.insert(0, r'$\text{Testset Accuracy on COM [mrad] := }\theta(\vec{r}_\text{G,COM}, \vec{r}_\text{P})$')
            all_errors = pd.concat([df['error'].astype(float) for df_ls in [df_list, df_list_2] for df in df_ls])
            
        else:
            all_errors = pd.concat([df['error'].astype(float) for df in df_list])
        min_val, max_val = all_errors.min(), all_errors.max()
        bins = int((max_val - min_val) / bin_width) + 1
        colors = [mcolors.to_hex(cmap(i % 10)) for i in range(len(df_list))]
        colors.insert(0, colors.pop())
        
        if plot_title:
            axs[0].set_title(f"Tracking Errors over Testing Samples for '{scenario}'-Scenario", fontsize=14)
        axs[0].scatter([], [], label=r'$\text{Alignment-Loss}(\vec{r}_\text{G}, \vec{r}_\text{P})=\text{MSE}(1, \text{COS-SIM}(\vec{r}_\text{G}, \vec{r}_\text{P}))$', 
                       alpha=0.0)
        
        max_yaxis = 0
        for ax, df_ls in enumerate([df_list, df_list_2]):
            if df_ls is None:
                continue
    
            for idx, df in enumerate(df_ls):
                trained_on_ideal = True if idx == 0 else False  
                errors = df['error'].astype(float)
                label = labels[idx]
                color = colors[idx]
                
                mean_alpha = 0.6
                mean_linewidth = 3.5
                mean_val = errors.mean()
                label_mean = f"Mean: {mean_val:.2f} mrad"
                
                if not trained_on_ideal:
                    mean_linestyle = ['--', '-.', ':', '--', '-.', ':'][idx]
                    n, bins, patches = axs[ax].hist(errors, bins=bins, range=(0, max_val), alpha=0.5, 
                            edgecolor='black', label=label, color=color)
                else:
                    mean_linestyle = '-'
                    mean_alpha = 1.0
                    mean_linewidth = 1.5
                    if 'base' in scenario.lower():
                        n, bins, patches = axs[ax].hist(errors, bins=bins, range=(0, max_val), alpha=0.5, 
                                    edgecolor='black', label=label, color=color)
                        for patch in patches:
                            patch.set_hatch('//')
                    else:
                        label_mean = label[0] + f" Mean: {mean_val:.2f} mrad"
                
                max_yaxis = n.max() if n.max() > max_yaxis else max_yaxis
                axs[ax].axvline(mean_val, color=color, linestyle=mean_linestyle, linewidth=mean_linewidth, alpha=mean_alpha, label=label_mean)
            
            axs[ax].set_ylabel('Absolute Frequency', fontsize=11)
            axs[ax].legend()
            axs[ax].grid(True, linestyle="--", alpha=0.5)
            axs[ax].set_xlabel(x_labels[ax], fontsize=11)
        
        for ax in axs:    
            ax.set_ylim(0, max_yaxis + 10)

        plt.tight_layout()
        plt.savefig(output_path, format="pdf")
        plt.close()
        print(f"Saved dual-distribution histogram to: {output_path}")


def plot_grid_of_error_distributions(scenario_grid: List[List[Optional[Tuple[List[pd.DataFrame], str]]]],
                                     calibration_labels: List[str], output_path: str, bin_width: float = 0.1):
    """
    Generate a 3x3 grid of subplots to visualize alignment error distributions for different calibration scenarios.

    Parameters
    ----------
    scenario_grid : 3x3 List of lists
        Each element is either None or a tuple (list of 3 pd.DataFrame for each strategy, scenario title).
    calibration_labels : List[str]
        Names for the 3 calibration strategies, used in shared legend.
    output_path : str
        Path to save the resulting figure.
    bin_width : float
        Width of histogram bins in mrad.
    """
    assert len(calibration_labels) == 3, "This function supports exactly 3 calibration strategies per subplot."

    # Colors for the 3 calibration strategies
    cmap = cm.get_cmap('Set1')
    colors = [mcolors.to_hex(cmap(i)) for i in range(3)]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=True)
    all_errors = []

    # Collect all error values for consistent binning
    for row in scenario_grid:
        for cell in row:
            if cell is not None:
                dfs, _ = cell
                for df in dfs:
                    all_errors.append(df["error"].astype(float))
    all_errors_concat = pd.concat(all_errors)
    min_val, max_val = all_errors_concat.min(), all_errors_concat.max()
    bins = int((max_val - min_val) / bin_width) + 1

    # Plot subfigures
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            cell = scenario_grid[i][j]

            if cell is None:
                ax.axis('off')
                continue

            dfs, title = cell

            for idx, df in enumerate(dfs):
                errors = df['error'].astype(float)
                label = calibration_labels[idx]
                color = colors[idx]

                ax.hist(errors, bins=bins, range=(min_val, max_val),
                        alpha=0.5, edgecolor='black', color=color)

                mean_val = errors.mean()
                ax.axvline(mean_val, color=color, linestyle='--', linewidth=1.5,
                           label=f"{label} Mean ({mean_val:.2f} mrad)")

            ax.set_title(title, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)

            if i == 2:
                ax.set_xlabel("Tracking Error [mrad]", fontsize=10)
            if j == 0:
                ax.set_ylabel("Frequency", fontsize=10)

    # Shared legend in top right subplot (1st row, 3rd column)
    handles = [plt.Line2D([0], [0], color=colors[i], lw=4, label=calibration_labels[i])
               for i in range(3)]
    axes[0][2].legend(handles=handles, loc='upper right', fontsize=10)
    axes[0][2].set_axis_off()

    plt.tight_layout()
    plt.savefig(output_path, format="pdf")
    plt.close()
    print(f"Saved grid of error distributions to: {output_path}")
    

def plot_error_boxplots_by_scenario_old(
    scenario_dfs: List[pd.DataFrame],
    scenario_labels: List[str],
    output_path: str,
    y_label: str = "Tracking Error [mrad]",
    plot_title: str = "Tracking Error Distribution by Blocking Scenario"
):
    """
    Generate a boxplot summary of alignment error distributions for multiple calibration scenarios.

    Parameters
    ----------
    scenario_dfs : List[pd.DataFrame]
        One DataFrame per scenario, each containing a column 'error' with tracking errors in mrad.
    scenario_labels : List[str]
        Names of the calibration scenarios shown on the x-axis.
    output_path : str
        Path to save the plot as a PDF.
    y_label : str
        Label for the y-axis.
    plot_title : str
        Title of the plot.
    """
    assert len(scenario_dfs) == len(scenario_labels), "Mismatch between scenarios and labels."

    # Extract error data from each scenario
    data = [df["error"].astype(float).values for df in scenario_dfs]

    fig, ax = plt.subplots(figsize=(12, 6))

    box = ax.boxplot(
        data,
        vert=True,
        patch_artist=True,
        tick_labels=scenario_labels,
        showmeans=True,
        showfliers=False,       # show all values outside the box
        whis=[0,100],                # no whiskers beyond Q1–Q3
        flierprops=dict(marker='o', color='red', markersize=4, alpha=0.6),
        boxprops=dict(facecolor="lightblue", color="blue"),
        capprops=dict(color="blue"),
        whiskerprops=dict(color="blue"),
        medianprops=dict(color="black"),
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='orange')
    )

    # Add legend once for plot features
    ax.plot([], [], color='black', label='Median')
    ax.plot([], [], marker='D', color='orange', linestyle='None', label='Mean')
    ax.plot([], [], color='blue', linestyle='-', label='Range (Min/Max)')
    ax.plot([], [], marker='s', color='lightblue', linestyle='None', label='Interquartile Range')

    ax.legend(loc='upper center', fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    ax.set_title(plot_title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, format="pdf")
    plt.close()
    print(f"Saved boxplot to: {output_path}")


def plot_error_boxplots_by_scenario(
    scenario_dfs: List[pd.DataFrame],
    scenario_labels: List[str],
    output_path: str,
    y_label: str = "Tracking Error [mrad]",
    plot_title: str = "Tracking Error Distribution by Blocking Scenario",
    use_violin: bool = False,
    benchmark = None
):
    """
    Generate a boxplot or violin plot summary of alignment error distributions 
    for multiple calibration scenarios.

    Parameters
    ----------
    scenario_dfs : List[pd.DataFrame]
        One DataFrame per scenario, each containing a column 'error' with tracking errors in mrad.
    scenario_labels : List[str]
        Names of the calibration scenarios shown on the x-axis.
    output_path : str
        Path to save the plot as a PDF.
    y_label : str
        Label for the y-axis.
    plot_title : str
        Title of the plot.
    use_violin : bool
        If True, generate violin plots instead of boxplots.
    """
    assert len(scenario_dfs) == len(scenario_labels), "Mismatch between scenarios and labels."

    if use_violin:
        # Combine all data into a single DataFrame
        combined_df = pd.DataFrame()
        for df, label in zip(scenario_dfs, scenario_labels):
            temp_df = df.copy()
            temp_df["scenario"] = label
            combined_df = pd.concat([combined_df, temp_df], axis=0)

        combined_df["error"] = combined_df["error"].astype(float)
        all_max = combined_df["error"].max()
        
        fig, ax = plt.subplots(figsize=(12, all_max/2))
        sns.violinplot(
            x="scenario",
            y="error",
            data=combined_df,
            ax=ax,
            palette=["lightblue"],
            linewidth=1.2,
            inner=None,
            cut=0.0
        )

        ax.text(3, all_max +2.0, plot_title, ha='center', va='bottom', fontsize=15, color='black')
        
        if benchmark is not None:
            ax.axhline(y=benchmark, color='red', linestyle=':', linewidth=1.5, label=rf"$\text{{Benchmark @ }}{benchmark:.2f}$") 
            
        # Overlay mean and median
        for i, label in enumerate(scenario_labels):
            errors = combined_df[combined_df["scenario"] == label]["error"]
            mean_val = errors.mean()
            median_val = errors.median()
            max_val = errors.max()
            if label == "60%_on_3rd":
                min, max = errors.min(), errors.max()
            ax.plot(i, mean_val, marker='D', color='orange', label='Mean' if i == 0 else "")
            ax.plot([i - 0.2, i + 0.2], [median_val, median_val], color='black', lw=2, label='Median' if i == 0 else "")
            
            # Add text with mean and max
            ax.text(i, all_max + 0.5, rf"$\mathrm{{mean}}={mean_val:.2f}$" + "\n" + rf"$\mathrm{{max}}={max_val:.2f}$", 
                    ha='center', va='bottom', fontsize=12, color='black')
            
        ax.legend(loc='upper center', fontsize=12)

    else:
        # Prepare boxplot data
        data = [df["error"].astype(float).values for df in scenario_dfs]
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.boxplot(
            data,
            vert=True,
            patch_artist=True,
            tick_labels=scenario_labels,
            showmeans=True,
            showfliers=False,
            whis=[0, 100],
            flierprops=dict(marker='o', color='red', markersize=4, alpha=0.6),
            boxprops=dict(facecolor="lightblue", color="blue"),
            capprops=dict(color="blue"),
            whiskerprops=dict(color="blue"),
            medianprops=dict(color="black"),
            meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='orange')
        )

        # Custom legend
        ax.plot([], [], color='black', label='Median')
        ax.plot([], [], marker='D', color='orange', linestyle='None', label='Mean')
        ax.plot([], [], color='blue', linestyle='-', label='Range (Min/Max)')
        ax.plot([], [], marker='s', color='lightblue', linestyle='None', label='Interquartile Range')
        ax.legend(loc='upper center', fontsize=12)

    # Shared styling
    ax.set_xlabel('', fontsize=1)
    ax.set_ylabel(y_label, fontsize=12)
    # if plot_title is not None:
    #     ax.set_title(plot_title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf")
    plt.close()
    print(f"Saved {'violin' if use_violin else 'box'} plot to: {output_path}")


if __name__ == '__main__':
    
    run = "/dss/dsshome1/05/di38kid/data/results/runs/run_2506302315_20_Heliostats"
    output_dir = f"{run}/plots"
    
    mode = "Test"
    df_avg_actual_test_errors = load_final_avg_alignment_errors( f"{run}/logs/ActualAlignmentErrors_mrad/Avg.csv", mode)
    df_avg_test_errors = load_final_avg_alignment_errors(f"{run}/logs/AlignmentErrors_mrad/Avg.csv", mode)
    
    tower_measurements = "/dss/dsshome1/05/di38kid/data/paint/WRI1030197-tower-measurements.json"
    tower_positions = load_tower_measurements(tower_measurements)
    
    heliostat_metadata = "/dss/dsshome1/05/di38kid/data/paint/metadata/deflectometry_metadata_all_heliostats.csv"
    heliostat_df = load_heliostat_positions(heliostat_metadata, tower_measurements)
    
    df_actual_errors = load_alignment_errors(f"{run}/logs/ActualAlignmentErrors_mrad.csv")
    calibration_metadata = "/dss/dsshome1/05/di38kid/data/paint/selected_20/metadata/calibration_metadata_selected_heliostats_20250525_161028.csv"
    sun_positions_df = load_sun_positions(calibration_metadata)
    merged_df = merge_data(df_actual_errors, sun_positions_df)
    df_avg_errors = load_avg_alignment_errors(f"{run}/logs/ActualAlignmentErrors_mrad/Avg.csv")
    
    """Generate the scatters plot for k-NN data splits per Heliostat over sun positions (2D)."""
    # plot_splits(df_actual_errors, output_dir, sun_positions_df)
    
    """Generate the scatter plots for errors (ideal) distributions per Heliostat over sun positions (2D)."""
    # plot_alignment_errors_over_sun_pos(merged_df, output_dir, type='show_size')
    
    """Generate the scatter plots for mean test errors per Heliostat over its position in field (Ideal vs. Non-Ideal)."""
    title = "Final Test Avg. Tracking Error using Contour Loss [Base]"
    save_under = "avg_test_error_field_distribution_actual_vs_measured"
    # plot_test_errors_comparison_over_field(df_avg_actual_test_errors, heliostat_df, title, output_dir, save_under,
    #                                        df_errors=df_avg_test_errors, 
    #                                        tower_positions=tower_positions)

    # Histogram directory
    histogram_dir = os.path.join(output_dir, "histograms")
    os.makedirs(histogram_dir, exist_ok=True)
    # plot_error_histograms_per_heliostat(df_actual_errors[df_actual_errors["mode"] == mode], histogram_dir)

    combined_histogram_path = os.path.join(histogram_dir, "combined_error_histogram.pdf")
    # plot_combined_error_histogram(df_actual_errors[df_actual_errors["mode"] == mode], combined_histogram_path)


    """
    Use multiple erorr distributions for comparison across runs.
    """     
                        # Geometric        Geometric w/ COM     Contour         
    all_result_runs = [['run_2507101421', 'run_2507101427', 'run_2506301357'],  # Base
                       ['run_2507111022', 'run_2507101255', 'run_2507101348'],  # Realistic
                       ['run_2507011659', 'run_2507011050', 'run_2507141711'], # 20%_4th
                       ['run_2507011657', 'run_2507011049', 'run_2506301937'], # 20%_3rd
                       ['run_2507021056', 'run_2507011055', 'run_2506301939'], # 40%_4th
                       ['run_2507021047', 'run_2507011057', 'run_2507011154'], # 40%_3rd
                       ['run_2507021041', 'run_2507011044', 'run_2506302315'], # 60%_4th
                       ['run_2507021045', 'run_2507011045', 'run_2506301944'], # 60%_3rd
    ]
    final_ouptut_dir = '/dss/dsshome1/05/di38kid/data/results/plots/final_result_plots'
    import copy
    names = copy.deepcopy(all_result_runs[0])
    labels = []
    scenario='Base'
    names.insert(0, 'run_2507101431')  # Geo w/ ideal for 'Base' Dataset
    labels.append(r'$\text{Alignment-Loss}(\vec{r}_\text{G}, {\vec{r}_\text{P}})$')
    
    actual_error_dataframes = []
    approx_error_dataframes = []
    for i, name in enumerate(names):
        run = f'/dss/dsshome1/05/di38kid/data/results/runs/{name}_20_Heliostats'
        idx=-1
        if i == 3:
            idx=-2
        df_actual_errors = load_alignment_errors(f"{run}/logs/ActualAlignmentErrors_mrad.csv", idx=idx)
        actual_error_dataframes.append(df_actual_errors[df_actual_errors["mode"] == mode])
        df_errors = load_alignment_errors(f"{run}/logs/AlignmentErrors_mrad.csv", idx=idx)
        approx_error_dataframes.append(df_errors[df_errors["mode"] == mode])
    
    # labels = ['Conventional Geometric', 'Conventional Geometric w/ COM', 'Image-Based / Contour', # 'Geometric w/ ideal RA'
    #           ]
    
    alignment_loss = r'$\text{Alignment-Loss}=\text{MSE}(1, \text{COS-SIM}(\vec{r}_\text{G}, {\vec{r}_{\text{P}}}))$'
    labels.append(r'$\text{Alignment-Loss}(\vec{r}_\text{G,COM}, {\vec{r}_{\text{P}}})$')
    labels.append(r'$\text{Alignment-Loss}(\vec{r}_{\text{G,COM}}, {\vec{r}_{\text{P,COM}}})$')
    labels.append(r'$\text{Contour-Loss}(I_{\text{G}}, I_{\text{P}})$')
    
    final_hist_dir = f'{final_ouptut_dir}/histograms_meta'
    os.makedirs(final_hist_dir, exist_ok=True)
    plot_multiple_error_distributions(approx_error_dataframes, labels, f'{final_hist_dir}/250711_{scenario}_with_mean.pdf', scenario=scenario, plot_title=False,
                                      df_list_2=actual_error_dataframes
                                      )
    
    sys.exit()
    
    error_dataframes = []
    for i, scenario in enumerate(all_result_runs):
        run = f'/dss/dsshome1/05/di38kid/data/results/runs/{scenario[1]}_20_Heliostats'
        idx = -1
        if i == 0:
            idx = -1
        if i == 1:
            continue # do not include realistic scenario
        df_actual_errors = load_alignment_errors(f"{run}/logs/ActualAlignmentErrors_mrad.csv", idx=idx)
        error_dataframes.append(df_actual_errors[df_actual_errors["mode"] == mode])
    
    meta='geo_com'
    plot_error_boxplots_by_scenario(error_dataframes, ['Base', '20%_on_4th', '20%_on_3rd', '40%_on_4th', '40%_on_3rd', '60%_on_4th', '60%_on_3rd'], 
                                    f'{final_hist_dir}/250717_{meta}_violinplot.pdf', use_violin=True, plot_title=labels[2], benchmark=0.34)
    
    sys.exit() 
    
    """
    Show gird of erorr distributions for comparison over all scenarios across runs.
    """   
    all_error_dfs = []
    for scenario_runs in all_result_runs:
        all_error_dfs.append([])
        for run_name in scenario_runs:
            run_path = f'/dss/dsshome1/05/di38kid/data/results/runs/{run_name}_20_Heliostats'
            # df_actual_errors = load_alignment_errors(f"{run_path}/logs/ActualAlignmentErrors_mrad.csv")
            # all_error_dfs[-1].append(df_actual_errors[df_actual_errors["mode"] == mode])

    grid_error_dfs = [[None,                            (all_error_dfs[0], 'Base'),         None,],
                      [(all_error_dfs[1], '20_on_4th'), (all_error_dfs[2], '40_on_4th'),    (all_error_dfs[3], '60_on_4th'),],
                      [(all_error_dfs[4], '20_on_3rd'), (all_error_dfs[5], '40_on_3rd'),    (all_error_dfs[6], '60_on_3rd'),],]
    
    # plot_grid_of_error_distributions(grid_error_dfs, labels, f'{final_hist_dir}/meta_with_mean.pdf')
