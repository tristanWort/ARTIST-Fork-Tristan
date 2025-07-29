import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Literal, List, Dict
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_sun_positions(metadata_file):
    """
    Load sun position data from a CSV file
    
    Expected format: calib_id,azimuth,elevation
    
    Args:
        metadata_file: Path to CSV file with sun position data
    
    Returns:
        DataFrame with sun position data
    """
    # Load the data
    sun_data = pd.read_csv(metadata_file)
    
    # Ensure the dataframe has the required columns
    required_cols = ['Id', 'Azimuth', 'Elevation']
    for col in required_cols:
        if col not in sun_data.columns:
            raise ValueError(f"Sun positions file must contain column: {col}")
        
    sun_data = sun_data.rename(columns={'Id':'calib_id',
                                        'Azimuth': 'azimuth',
                                        'Elevation': 'elevation'})
    
    return sun_data

def merge_data(alignment_errors_df, sun_positions_df):
    """
    Merge alignment errors with sun position data
    
    Args:
        alignment_errors_df: DataFrame with alignment errors
        sun_positions_df: DataFrame with sun position data
    
    Returns:
        DataFrame with merged data
    """
    alignment_df = alignment_errors_df.copy()
    sun_df = sun_positions_df.copy()
    
    alignment_df['calib_id'] = alignment_df['calib_id'].astype(str)
    sun_df['calib_id'] = sun_df['calib_id'].astype(str)
    
    # Merge the dataframes on calibration ID
    merged_df = alignment_df.merge(
        sun_df,
        on='calib_id',
        how='inner'
    )
    
    # If no matches, print warning
    if len(merged_df) == 0:
        print("Warning: No matching calibration IDs found between alignment errors and sun positions")
    
    return merged_df


def plot_alignment_errors_over_sun_pos(merged_data, 
                                       output_dir,
                                       error_label='Tracking Error',
                                       sep_plots: bool=True,
                                       print_mean_error: bool=True,
                                       marker_for_training='*'):
    """
    Generate plots of tracking errors over sun position for each heliostat.
    Also, generates a plot displaying all errors for all heliostats.
    
    Args:
        merged_data: DataFrame with merged alignment errors and sun positions
        output_dir: Directory to save plots
        sep_plots: If True, then errors for will be shown per heliostat.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique heliostat IDs
    heliostat_ids = merged_data['heliostat_id'].unique()
    max_error = merged_data["error"].max()
    
    markers = {'Training': marker_for_training, 'Validation': 'o', 'Testing': 'o'}
    cmap = {'Training': 'black', 'Validation': 'blue', 'Testing': 'red'}
    if marker_for_training == 'o':  # will display training error as scaled ring
        cmap['Training'] = 'green'
    marker_scaling = 10.0  # marker_size = marker_scaling * error
    
    # Helper function to plot one figure
    def plot_for_data(data, heliostat_label, filename):
        fig, ax = plt.subplots(figsize=(10, 8))
        modes = data['mode'].unique()
        for mode in modes:
            mode_data = data[data['mode'] == mode]
            if mode_data.empty:
                continue

            # Plot data points
            for _, row in mode_data.iterrows():
                if mode == 'train' and marker_for_training == '*':
                    ax.plot(row['azimuth'], row['elevation'],
                            color=cmap[mode],
                            marker=markers[mode],
                            markersize=0.5)
                    continue

                ring_radius = 1.0
                while ring_radius <= (row['error'] + 1.0):
                    ax.plot(row['azimuth'], row['elevation'],
                            markersize=ring_radius * marker_scaling,
                            color='grey', fillstyle='none', marker='o', alpha=0.4)
                    ring_radius += 1.0

                ax.plot(row['azimuth'], row['elevation'],
                        color=cmap[mode],
                        marker=markers[mode],
                        markersize=row['error'] * marker_scaling,
                        alpha=0.4)

            # Compute mean error and add to legend
            label = mode
            if print_mean_error:
                mean_error = mode_data['error'].mean()
                label = f"{mode} [{round(mean_error, 2)} mrad]"
            ax.scatter([], [], s=1.0 * marker_scaling, marker=markers[mode],
                      facecolors='none', edgecolors=cmap[mode], label=label)
        
        ax.legend(loc='lower right', fontsize=11)
        az_max = data["azimuth"].max()
        ax.plot(77, 9, color='grey', fillstyle='none', marker='o', markersize=1 * marker_scaling, alpha=0.4)
        ax.text(80, 8.5, '= 1 mrad', fontsize=11)

        ax.set_xlabel('Sun Azimuth [°]')
        ax.set_ylabel('Sun Elevation [°]')
        ax.set_title(f"{error_label} for {heliostat_label}")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), format="pdf")
        plt.close(fig)
            
    # Plot per heliostat
    if sep_plots:
        for heliostat_id in heliostat_ids:
            heliostat_data = merged_data[merged_data['heliostat_id'] == heliostat_id]
            plot_for_data(heliostat_data, f"Heliostat {heliostat_id}", f"{heliostat_id}_alignment_errors.pdf")
            print(f"Plot for heliostat {heliostat_id} saved to {output_dir}")

    # Combined plot for all heliostats
    plot_for_data(merged_data, "Scenario '20%_on_3rd'", "all_heliostats_alignment_errors.pdf")
    print(f"Combined plot for all heliostats saved to {output_dir}")


def plot_errors_over_heliostats_by_distance(
    merged_data: pd.DataFrame,
    heliostat_distances: Dict[str, float],
    output_path: str,
    title = None,
    print_mean_error: bool = True,
    marker_for_training: str = '*',
):
    """
    Plot errors as marker sizes over heliostat IDs (sorted by distance), with sun azimuth on the y-axis.

    Parameters
    ----------
    merged_data : pd.DataFrame
        DataFrame with columns: ['heliostat_id', 'azimuth', 'error', 'mode']
    heliostat_distances : Dict[str, float]
        Mapping from heliostat_id to distance from power plant [m]
    output_path : str
        Path to save the plot as a PDF
    error_label : str
        Label for the plot title
    print_mean_error : bool
        Whether to print mean error in the legend
    marker_for_training : str
        Marker for training samples
    """
    # Validate input
    assert 'heliostat_id' in merged_data.columns
    assert 'azimuth' in merged_data.columns
    assert 'error' in merged_data.columns
    assert 'mode' in merged_data.columns

    # Filter out heliostats not in distance dict
    merged_data = merged_data[merged_data['heliostat_id'].isin(heliostat_distances.keys())]

    # Create x-ticks: Heliostat ID (distance)
    unique_ids = sorted(set(merged_data['heliostat_id']), key=lambda h: heliostat_distances[h])
    xtick_labels = [f"{hid} ({heliostat_distances[hid]:.1f} m)" for hid in unique_ids]
    id_to_xtick = {hid: idx for idx, hid in enumerate(unique_ids)}

    markers = {'Training': marker_for_training, 'Validation': 'o', 'Testing': 'o'}
    cmap = {'Training': 'black', 'Validation': 'blue', 'Testing': 'red'}
    if marker_for_training == 'o':
        cmap['Training'] = 'green'
    marker_scaling = 10.0

    # Start plotting
    fig, ax = plt.subplots(figsize=(max(12, len(unique_ids) * 0.6), 14))
    modes = merged_data['mode'].unique()

    # Add reference marker
    ax.scatter([], [], s=(marker_scaling)**2, marker='o', color='grey', facecolors='none', label='= 1.0 mrad')
    
    for mode in modes:
        mode_data = merged_data[merged_data['mode'] == mode]
        if mode_data.empty:
            continue

        for _, row in mode_data.iterrows():
            x_pos = id_to_xtick[row['heliostat_id']]
            y_pos = row['elevation']
            error = row['error']
            if mode == 'Training' and marker_for_training == '*':
                ax.plot(x_pos, y_pos, marker='*', color=cmap[mode], markersize=4)
            else:
                ring_radius = 1.0
                while ring_radius <= (error + 1.0):
                    ax.plot(x_pos, y_pos,
                            markersize=ring_radius * marker_scaling,
                            color='grey', fillstyle='none', marker='o', alpha=0.4)
                    ring_radius += 1.0

                ax.plot(x_pos, y_pos,
                        marker=markers[mode],
                        color=cmap[mode],
                        markersize=error * marker_scaling,
                        alpha=0.4)

        # Legend: dummy marker with mean error
        label = mode
        if print_mean_error:
            mean_error = mode_data['error'].mean()
            label = f"{mode} [{mean_error:.2f} mrad]"
        ax.scatter([], [], s=(marker_scaling)**2, marker=markers[mode], color=cmap[mode], label=label, alpha=0.4)
    
    # ax.plot(15.3, merged_data['elevation'].max() +6, color='grey', fillstyle='none',
    #         marker='o', markersize=1 * marker_scaling)
    # ax.text(15.5, merged_data['elevation'].max() +5.5, '= 1 mrad', fontsize=10)

    # Axis labels
    ax.set_xlabel("Heliostat ID (Distance from Power Plant)", fontsize=24, labelpad=8)
    ax.set_ylabel("Sun Elevation [°]", fontsize=24)
    if title is not None:
        ax.set_title(f"{title}", fontsize=20)
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, rotation=55, ha='right', fontsize=20)
    ax.set_ylim(-2, merged_data['elevation'].max()+2)
                # merged_data['elevation'].max() + 15)
    ax.tick_params(axis='y', labelsize=24)
    ax.legend(loc='lower center', fontsize=20, ncol=len(ax.get_legend_handles_labels()[0]))
    ax.grid(True, linestyle='--', alpha=0.6)

    # Save
    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close()
    print(f"Saved error overview plot to: {output_path}")


def plot_error_kde_bands_over_heliostats_by_distance(
    merged_data: pd.DataFrame,
    heliostat_distances: Dict[str, float],
    output_path: str,
    title=None,
    print_mean_error: bool = True,
    marker_for_training: str = '*',
    bandwidth: float = 1.0,
    elevation_res: int = 200
):
    """
    Plot errors as KDE-filled vertical bands per heliostat and mode, using error magnitude as horizontal width.

    Parameters
    ----------
    merged_data : pd.DataFrame
        Columns: ['heliostat_id', 'elevation', 'error', 'mode']
    heliostat_distances : Dict[str, float]
        Maps heliostat_id -> distance in [m]
    output_path : str
        Where to save the output figure (PDF)
    title : str, optional
        Title for the figure
    print_mean_error : bool
        Whether to include mean error in legend
    marker_for_training : str
        For legend consistency, not used in plot
    bandwidth : float
        Bandwidth used for KDE smoothing
    elevation_res : int
        Number of vertical points to sample for KDE line
    """
    from scipy.stats import gaussian_kde
    
    assert {'heliostat_id', 'elevation', 'error', 'mode'}.issubset(merged_data.columns)

    # Filter valid heliostats
    merged_data = merged_data[merged_data['heliostat_id'].isin(heliostat_distances.keys())]

    # Order heliostats by distance
    unique_ids = sorted(set(merged_data['heliostat_id']), key=lambda h: heliostat_distances[h])
    xtick_labels = [f"{hid} ({heliostat_distances[hid]:.1f} m)" for hid in unique_ids]
    id_to_xtick = {hid: idx for idx, hid in enumerate(unique_ids)}

    # Color per mode
    modes = ['Training', 'Validation', 'Testing']
    colors = {'Training': 'black', 'Validation': 'blue', 'Testing': 'red'}

    fig, ax = plt.subplots(figsize=(max(12, len(unique_ids) * 0.6), 8))

    elevation_min = merged_data['elevation'].min() - 5
    elevation_max = merged_data['elevation'].max() + 15
    elevation_grid = np.linspace(elevation_min, elevation_max, elevation_res)

    # Determine global max error for consistent scaling
    global_max_error = merged_data['error'].max()
    width_scale = 0.4  # max width in "x-axis units" per 1.0 mrad

    for mode in modes:
        mode_data = merged_data[merged_data['mode'] == mode]
        if mode_data.empty:
            continue

        for heliostat_id in unique_ids:
            data = mode_data[mode_data['heliostat_id'] == heliostat_id]
            if data.empty:
                continue

            x_center = id_to_xtick[heliostat_id]
            elevations = data['elevation'].values
            errors = data['error'].values

            # KDE approximation of error as a function of elevation
            try:
                kde = gaussian_kde(elevations, weights=errors, bw_method=bandwidth / np.std(elevations))
                error_profile = kde(elevation_grid)
            except Exception:
                continue  # fallback if KDE fails due to few points

            # Normalize error profile using global max error
            widths = (error_profile / global_max_error) * width_scale

            ax.fill_betweenx(
                y=elevation_grid,
                x1=x_center - widths,
                x2=x_center + widths,
                color=colors[mode],
                alpha=0.3,
                linewidth=0
            )

        # Add dummy for legend
        label = mode
        if print_mean_error:
            mean_error = mode_data['error'].mean()
            label = f"{mode} [{mean_error:.2f} mrad]"
        ax.plot([], [], color=colors[mode], alpha=0.5, linewidth=10, label=label)

    # Legend: width = 1 mrad reference band
    ax.fill_betweenx(
        y=[elevation_min + 2, elevation_min + 10],
        x1=[-width_scale] * 2,
        x2=[+width_scale] * 2,
        color='gray',
        alpha=0.3,
        label='= 1.0 mrad'
    )

    # Labels and layout
    ax.set_xlabel("Heliostat ID (Distance from Power Plant)", fontsize=20, labelpad=8)
    ax.set_ylabel("Sun Elevation [°]", fontsize=20)
    if title:
        ax.set_title(title, fontsize=20)
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(elevation_min, elevation_max)
    ax.legend(loc='upper center', fontsize=18, ncol=len(ax.get_legend_handles_labels()[0]))
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, format='pdf')
    plt.close()
    print(f"Saved normalized KDE band plot to: {output_path}")


def plot_error_violinplots_by_scenario(scenario_dfs: List[pd.DataFrame], scenario_labels: List[str], output_dir: str, 
                                       y_label: str = "Tracking Error [mrad]", plot_title: str = "Tracking Error Distribution by Scenario"):
    """
    Generate a violin plot of alignment error distributions for multiple calibration scenarios.

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
    assert len(scenario_dfs) == len(scenario_labels), "Mismatch between dataframes and labels."
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a combined DataFrame for plotting
    combined_df = pd.DataFrame()
    for df, label in zip(scenario_dfs, scenario_labels):
        temp_df = df.copy()
        temp_df["scenario"] = label
        combined_df = pd.concat([combined_df, temp_df], axis=0)

    # Ensure proper types
    combined_df["error"] = combined_df["error"].astype(float)
    combined_df["scenario"] = combined_df["scenario"].astype(str)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.violinplot(
        x="scenario",
        y="error",
        data=combined_df,
        palette=['lightblue'],
        ax=ax,
        inner=None,          # Disable default inner representation
        linewidth=1.2,
        cut=0
    )

    # Overlay mean and median
    for i, scenario in enumerate(scenario_labels):
        errors = combined_df[combined_df["scenario"] == scenario]["error"]
        mean_error = errors.mean()
        median_error = errors.median()
        max_error = errors.max()

        # Plot mean as diamond
        ax.plot(i, mean_error, marker='D', color='orange', markersize=7, label='Mean' if i == 0 else "")
        # Plot median as horizontal black line
        ax.plot([i - 0.2, i + 0.2], [median_error, median_error], color='black', lw=2, label='Median' if i == 0 else "")

        # Add text with mean and max at y = 10 mrad
        ax.text(i, max_error +1.5, rf"$\mathrm{{mean}}={mean_error:.2f}$" + "\n" + rf"$\mathrm{{max}}={max_error:.2f}$", 
                ha='center', va='bottom', fontsize=12, color='black')
        
    # Styling
    ax.set_xlabel('', fontsize=0)
    ax.set_ylabel(y_label, fontsize=12)
    if plot_title is not None:
        ax.set_title(plot_title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticklabels(scenario_labels, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Add legend once
    ax.legend(loc='upper center', fontsize=12)

    # Save
    plt.tight_layout()
    filename = "all_reflection_axis_violinplot.pdf"
    fig.savefig(os.path.join(output_dir, filename), format="pdf")
    plt.close(fig)
    print(f"Saved violin plot to: {output_dir}")


def plot_alignment_errors_over_sun_pos_old(merged_data, 
                          output_dir, 
                          average_errors=None,
                          type=Literal['show_size', 'show_color_gradient'], 
                          interpolate=False, 
                          grid_resolution=100):
    """
    Generate plots of alignment errors vs sun position for each heliostat
    
    Args:
        merged_data: DataFrame with merged alignment errors and sun positions
        output_dir: Directory to save plots
        type: How to display the errors
        interpolate: Whether to interpolate the data for a smoother plot
        grid_resolution: Resolution of the interpolation grid
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique heliostat IDs
    heliostat_ids = merged_data['heliostat_id'].unique()
    
    for heliostat_id in heliostat_ids:
        # Filter data for this heliostat
        heliostat_data = merged_data[merged_data['heliostat_id'] == heliostat_id]
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique modes for this heliostat
        modes = heliostat_data['mode'].unique()
        
        min_error = heliostat_data['error'].min()
        max_error = heliostat_data['error'].max()
        
        if type == 'show_color_gradient':
            markers = {'Training': 'o', 'Validation': 's', 'Testing': '^'}
            # Calculate global min/max for colormap normalization
            # Define markers for different mode
            norm = colors.Normalize(vmin=min_error, vmax=max_error)
            cmap = plt.cm.viridis
        
        elif type == 'show_size':
            markers = {'Training': '*', 'Validation': 'o', 'Testing': 'o'}
            cmap = {'Training': 'black', 'Validation': 'blue', 'Testing': 'red'}
            marker_scaling = 20
        
        # For scatter plots or interpolation points
        for mode in modes:
            mode_data = heliostat_data[heliostat_data['mode'] == mode]
            # Define markers for different mode
            
            if not mode_data.empty:
                
                if type=='show_size':                
                    for i, data in mode_data.iterrows():  
                               
                        # if mode == 'train':
                        #     # Plot black star for training samples
                        #     ax.plot(data['azimuth'],
                        #             data['elevation'],
                        #             color=cmap[mode],
                        #             marker=markers[mode],
                        #             markersize=10,
                        #             )  
                        #     continue  # Skip the rest for Train mode
                                    
                        # Plot empty circles around data points
                        plot_error = 1.0
                        while plot_error <= (data['error'] + 1.0):
                            ax.plot(data['azimuth'],
                                    data['elevation'], 
                                    color='grey', 
                                    fillstyle='none',
                                    marker='o', 
                                    markersize=plot_error * marker_scaling,
                                    )
                            plot_error += 1.0
                            
                        ax.plot(data['azimuth'],
                                data['elevation'],
                                color=cmap[mode],
                                marker=markers[mode],
                                markersize=data['error'] * marker_scaling,
                                alpha=0.6)
                        
                    
                elif type=='show_color_gradient':
                    scatter = ax.scatter(
                        mode_data['azimuth'],
                        mode_data['elevation'],
                        c=mode_data['error'],
                        cmap=cmap,
                        norm=norm,
                        marker=markers[mode],
                        s=100,
                        edgecolors='black',
                        linewidths=1,
                        alpha=0.8,
                        label=mode
                    )
                    
        
        # Interpolate if requested
        if interpolate and len(heliostat_data) > 3:  # Need at least 4 points for interpolation
            raise NotImplementedError('Interpoaltion not yet implemented!')
            # Create a regular grid
            x_min, x_max = heliostat_data['azimuth'].min(), heliostat_data['azimuth'].max()
            y_min, y_max = heliostat_data['elevation'].min(), heliostat_data['elevation'].max()
            
            # Add some padding
            x_padding = (x_max - x_min) * 0.05
            y_padding = (y_max - y_min) * 0.05
            
            x_min -= x_padding
            x_max += x_padding
            y_min -= y_padding
            y_max += y_padding
            
            xi = np.linspace(x_min, x_max, grid_resolution)
            yi = np.linspace(y_min, y_max, grid_resolution)
            xi, yi = np.meshgrid(xi, yi)
            
            # Interpolate
            points = heliostat_data[['azimuth', 'elevation']].values
            values = heliostat_data['error'].values
            
            zi = griddata(points, values, (xi, yi), method='cubic')
            
            # Plot the interpolated surface
            contour = ax.contourf(
                xi, yi, zi, 
                levels=50,
                cmap=cmap, 
                norm=norm,
                alpha=0.5
            )
        
        # Add colorbar
        if type=='show_color_gradient':
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            cbar.set_label('Alignment Error (mrad)')
        
        elif type=='show_size':
            ax.scatter([], [], s=marker_scaling, marker='o', facecolors='none', edgecolors='grey', label='1 mrad')
            for mode in modes:
                ax.scatter([], [], marker='o', c=cmap[mode], label=mode)
            
        # Add labels and title
        ax.set_xlabel('Sun Azimuth (degrees)')
        ax.set_ylabel('Sun Elevation (degrees)')
        title = f'Errors for Heliostat {heliostat_id} '
        if average_errors is not None:
            for mode, error in average_errors[heliostat_id].items():
                rounded_error = round(error['error'], 4)
                title += str(f'{mode} Avg: {rounded_error} mrad ')
        ax.set_title(title)
        
        # Add legend for modes
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{heliostat_id}_alignment_errors.png'), dpi=300)
        plt.close(fig)
        
        print(f"Plot for heliostat {heliostat_id} saved to {output_dir}")
   
def plot_error_histograms(merged_data, output_dir, bin_width=0.1):
    
    """
    Generate histogram plots of alignment error distributions for each heliostat.
    
    Args:
        merged_data: DataFrame with merged alignment errors and sun positions
        output_dir: Directory to save plots
        bin_width: Width of histogram bins in mrad (default: 0.1)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique heliostat IDs
    heliostat_ids = merged_data['heliostat_id'].unique()
    
    # Define colors for different modes
    mode_colors = {'Train': 'green', 'Validation': 'blue', 'Test': 'red'}
    
    for heliostat_id in heliostat_ids:
        # Filter data for this heliostat
        heliostat_data = merged_data[merged_data['heliostat_id'] == heliostat_id]
        
        # Get unique modes for this heliostat
        modes = heliostat_data['mode'].unique()
        
        # Calculate global min/max for all modes to ensure consistent x-axis
        min_error = heliostat_data['error'].min()
        max_error = heliostat_data['error'].max()
        
        # Add a small buffer to the max value to ensure all points are visible
        max_error = max_error * 1.05
        
        # Calculate number of bins based on range and bin width
        num_bins = int(np.ceil((max_error - min_error) / bin_width))
        
        # Create bins with specified width
        bins = np.arange(min_error, max_error + bin_width, bin_width)
        
        # Create figure with subplots (one row per mode)
        # Make the figure wider to accommodate the statistics table on the right
        fig, axs = plt.subplots(len(modes), 1, figsize=(16, 3 * len(modes)), sharex=True)
        
        # If there's only one mode, axs will not be an array
        if len(modes) == 1:
            axs = [axs]
        
        # Plot histogram for each mode
        for i, mode in enumerate(modes):
            mode_data = heliostat_data[heliostat_data['mode'] == mode]
            
            if not mode_data.empty:
                # Plot histogram
                counts, edges, bars = axs[i].hist(
                    mode_data['error'], 
                    bins=bins,
                    alpha=0.7,
                    color=mode_colors.get(mode, 'gray'),
                    edgecolor='black',
                    linewidth=1
                )
                
                # Add value labels above each bar
                for j, (count, x) in enumerate(zip(counts, edges[:-1])):
                    if count > 0:  # Only add label if bar has data
                        axs[i].text(
                            x + bin_width/2,  # Center of the bar
                            count + 0.1,      # Slightly above the bar
                            str(int(count)),  # Integer count
                            ha='center',
                            va='bottom',
                            fontsize=8
                        )
                
                # Calculate statistics for this mode
                mean_error = mode_data['error'].mean()
                median_error = mode_data['error'].median()
                std_dev = mode_data['error'].std()
                
                # Add vertical lines for mean and median
                axs[i].axvline(mean_error, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_error:.4f} mrad')
                axs[i].axvline(median_error, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_error:.4f} mrad')
                
                # Add vertical dotted line at mean + 2 standard deviations
                upper_bound = mean_error + 2 * std_dev
                axs[i].axvline(upper_bound, color='purple', linestyle=':', linewidth=1.5, 
                              label=f'Mean + 2σ: {upper_bound:.4f} mrad')
                
                # Add text with statistics - outside the plot area
                stats_text = (f"Statistics for {mode} mode:\n\n"
                              f"Count: {len(mode_data)}\n"
                              f"Mean: {mean_error:.4f} mrad\n"
                              f"Median: {median_error:.4f} mrad\n"
                              f"Std Dev: {std_dev:.4f} mrad\n"
                              f"Mean + 2σ: {upper_bound:.4f} mrad")
                
                # Create a text table outside the main plot area
                # Adjust the position if needed based on your preference
                props = dict(boxstyle='round', facecolor='white', alpha=0.9)
                
                # The coordinates are relative to the figure, not the axes
                # Position the text to the right of the plot
                fig.text(0.85, 0.85 - i * (1/len(modes)), stats_text, 
                        verticalalignment='top', horizontalalignment='center',
                        fontsize=10, bbox=props)
            
            # Set title and labels for each subplot
            axs[i].set_title(f'{mode} Errors')
            axs[i].set_ylabel('Frequency')
            axs[i].grid(True, linestyle='--', alpha=0.7)
            
            # Place legend inside the plot area in the upper left
            axs[i].legend(loc='upper right')
        
        # Set common x-label for all subplots
        axs[-1].set_xlabel('Alignment Error (mrad)')
        
        # Set common title for the figure
        fig.suptitle(f'Error Distribution for Heliostat {heliostat_id}', fontsize=16)
        
        # Adjust layout - use a tighter layout for the plots but leave room for stats table
        plt.subplots_adjust(right=0.75)  # Leave 25% of the figure width for the stats tables
        fig.tight_layout(rect=[0, 0, 0.75, 0.97])  # Leave space for the suptitle and stats tables
        
        # Save the figure
        fig_path = os.path.join(output_dir, f'{heliostat_id}_error_histogram.png')
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        
        print(f"Error histogram for heliostat {heliostat_id} saved to {fig_path}")

def analyze_heliostat_field(tensorboard_path, metadata_file, output_dir):
    """
    Main function to analyze and visualize heliostat field alignment errors
    
    Args:
        tensorboard_path: Path to tensorboard log directory
        metadata_file: Path to CSV file with sun position data
        output_dir: Directory to save plots
    """
    from extract_errors import get_all_alignment_errors, get_average_errors
    
    # Extract alignment errors from tensorboard
    print("Extracting alignment errors from tensorboard...")
    # heliostat_ids = ['AA39', 'AC27', 'AD43', 'AM35', 'BB72', 'BG24']
    alignment_errors = get_all_alignment_errors(tensorboard_path, 
                                                last_epoch_only=True,
                                                )

    print("Extracting average errors from tensorboard...")
    average_errors = dict()
    for heliostat_id in alignment_errors['heliostat_id'].unique():
        average_errors[heliostat_id] = get_average_errors(tensorboard_path, heliostat_id)
    
    # Load sun position data
    print("Loading sun position data...")
    sun_positions = load_sun_positions(metadata_file)
    
    # Merge the data
    print("Merging alignment errors with sun positions...")
    merged_data = merge_data(alignment_errors, sun_positions)
    
    # Generate plots
    print("Generating plots...")
    plot_alignment_errors_over_sun_pos(merged_data, output_dir, average_errors=average_errors, type='show_size', interpolate=False)
    
    plot_error_histograms(merged_data, output_dir, bin_width=0.1)
    
    print(f"Analysis complete. Plots saved to {output_dir}")
    
    # Return the merged data for further analysis if needed
    return merged_data

# Example usage:
if __name__ == "__main__":
    tensorboard_path = "/dss/dsshome1/05/di38kid/data/results/runs/run_2505150607_six_heliostats/logs/knn_(30, 30)"
    metadata_file = "/dss/dsshome1/05/di38kid/data/paint/metadata/calibration_metadata_selected_heliostats_20250325_150310.csv"
    output_dir = "/dss/dsshome1/05/di38kid/data/results/runs/run_2505150607_six_heliostats/plots"

    analyze_heliostat_field(tensorboard_path, metadata_file, output_dir)

    