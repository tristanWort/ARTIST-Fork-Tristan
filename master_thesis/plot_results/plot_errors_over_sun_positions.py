import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Literal
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

def plot_alignment_errors(merged_data, 
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
            markers = {'Train': 'o', 'Validation': 's', 'Test': '^'}
            # Calculate global min/max for colormap normalization
            # Define markers for different mode
            norm = colors.Normalize(vmin=min_error, vmax=max_error)
            cmap = plt.cm.viridis
        
        elif type == 'show_size':
            markers = {'Train': 'o', 'Validation': 'o', 'Test': 'o'}
            cmap = {'Train': 'green', 'Validation': 'blue', 'Test': 'red'}
            marker_scaling = 50 / max_error
        
        # For scatter plots or interpolation points
        for mode in modes:
            mode_data = heliostat_data[heliostat_data['mode'] == mode]
            # Define markers for different mode
            
            if not mode_data.empty:
                
                if type=='show_size':
                    for i, data in mode_data.iterrows():                        
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
                rounded_error = round(error['error'], 2)
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
    heliostat_ids = ['AA39', 'AC27', 'AD43', 'AM35', 'BB72', 'BG24']
    alignment_errors = get_all_alignment_errors(tensorboard_path, 
                                                last_epoch_only=True,
                                                heliostat_ids=heliostat_ids)

    print("Extracting average errors from tensorboard...")
    average_errors = dict()
    for heliostat_id in heliostat_ids:
        average_errors[heliostat_id] = get_average_errors(tensorboard_path, heliostat_id)
    
    # Load sun position data
    print("Loading sun position data...")
    sun_positions = load_sun_positions(metadata_file)
    
    # Merge the data
    print("Merging alignment errors with sun positions...")
    merged_data = merge_data(alignment_errors, sun_positions)
    
    # Generate plots
    print("Generating plots...")
    plot_alignment_errors(merged_data, output_dir, average_errors=average_errors, type='show_size', interpolate=False)
    
    print(f"Analysis complete. Plots saved to {output_dir}")
    
    # Return the merged data for further analysis if needed
    return merged_data

# Example usage:
if __name__ == "__main__":
    tensorboard_path = "/dss/dsshome1/05/di38kid/data/results/tensorboard/run_2504011544_AA39.AC27.AD43.AM35.BB72.BG24"
    metadata_file = "/dss/dsshome1/05/di38kid/data/paint/metadata/calibration_metadata_selected_heliostats_20250325_150310.csv"
    output_dir = "/dss/dsshome1/05/di38kid/data/results/plots/errors_over_sun_positions/run_2504011544_AA39.AC27.AD43.AM35.BB72.BG24"
    
    analyze_heliostat_field(tensorboard_path, metadata_file, output_dir)
    