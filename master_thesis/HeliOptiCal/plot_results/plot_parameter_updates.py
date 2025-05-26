import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing import event_accumulator
import glob
from collections import defaultdict
import seaborn as sns


def load_parameters_from_csv(csv_file):
    """
    Load parameter values from the CSV file with heliostat columns.
    
    Args:
        csv_file: Path to the CSV file containing parameter data
        
    Returns:
        modified_params: Dictionary mapping parameter names to their modified values per heliostat
        heliostat_ids: List of heliostat IDs from the CSV
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get heliostat IDs (all columns except the first one, which is 'Parameter')
    heliostat_ids = df.columns[1:].tolist()
    
    # Create a dictionary to store modified parameters
    modified_params = {}
    
    # Process each parameter (row)
    for _, row in df.iterrows():
        param_name = row['Parameter']
        
        # Create a dictionary mapping heliostat ID to its value for this parameter
        param_values = {heliostat_id: row[heliostat_id] for heliostat_id in heliostat_ids}
        
        # Store in the dictionary
        modified_params[param_name] = param_values
        
    return modified_params, heliostat_ids


def extract_tensorboard_parameters(tensorboard_log_dir):
    """
    Extract parameter evolution from Tensorboard logs with format: Parameters/param_name/heliostat_id
    
    Args:
        tensorboard_log_dir: Directory containing Tensorboard event files
        
    Returns:
        parameter_evolution: Dictionary mapping parameter names to heliostat IDs to epoch/value pairs
    """
    # Find all event files
    event_files = glob.glob(os.path.join(tensorboard_log_dir, "events.out.tfevents.*"))
    
    if not event_files:
        raise ValueError(f"No Tensorboard event files found in {tensorboard_log_dir}")
    
    # Initialize dictionary to store parameter evolution
    parameter_evolution = defaultdict(lambda: defaultdict(list))
    
    # Process each event file
    for event_file in event_files:
        # Load the event file
        ea = event_accumulator.EventAccumulator(
            event_file,
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.TENSORS: 0,
                event_accumulator.HISTOGRAMS: 0,
            }
        )
        ea.Reload()
        
        # First try to get scalar data (sometimes parameters are logged as scalars)
        for tag in ea.Tags()['scalars']:
            parts = tag.split('/')
            
            if len(parts) >= 3 and parts[0] == "Parameters":
                tb_param_name = parts[1]
                heliostat_id = parts[2]
                
                 # Handle actuator indices
                actuator_index = None
                if len(parts) >= 4 and ("actuators" in tb_param_name):
                    try:
                        actuator_index = int(parts[3]) + 1
                        # Use the index directly for the CSV format (e.g., actuators_offsets_0)
                        param_name = f"{tb_param_name}_{actuator_index}"
                    except (ValueError, IndexError):
                        # If we can't parse the index, use the original name
                        param_name = tb_param_name
                else:
                    # For non-actuator parameters, use the original name
                    param_name = tb_param_name
                
                print(f"Found scalar parameter: {param_name}, heliostat: {heliostat_id}")
                
                # Get scalar events for this tag
                events = ea.Scalars(tag)
                
                # Extract value from each event and append to list
                for event in events:
                    epoch = event.step
                    value = event.value
                    parameter_evolution[param_name][heliostat_id].append((epoch, value))
    
    # Check if we found any data
    if not parameter_evolution:
        print("WARNING: No parameter data was found in the TensorBoard logs!")
        
        # As a fallback, try to directly parse the event files to extract tensor values
        print("Attempting to directly parse event files for tensor values...")
        
        try:
            # Try to import tensorflow modules for direct parsing
            try:
                from tensorflow.python.summary.summary_iterator import summary_iterator
                from tensorflow.python.framework import tensor_util
                tf_available = True
            except ImportError:
                print("TensorFlow not available for direct parsing. Trying alternative method...")
                tf_available = False
            
            if tf_available:
                for event_file in event_files:
                    for event in summary_iterator(event_file):
                        for value in event.summary.value:
                            if "Parameters" in value.tag:
                                parts = value.tag.split('/')
                                if len(parts) >= 3 and parts[0] == "Parameters":
                                    param_name = parts[1]
                                    heliostat_id = parts[2]
                                    
                                    # Try to extract tensor value
                                    if value.HasField('tensor'):
                                        tensor = tensor_util.MakeNdarray(value.tensor)
                                        # For simplicity, if tensor is multi-dimensional, take the mean
                                        mean_value = float(np.mean(tensor))
                                        parameter_evolution[param_name][heliostat_id].append((event.step, mean_value))
                                        print(f"Extracted tensor value for {param_name}, {heliostat_id}: {mean_value}")
                    
                    
        except Exception as e:
            print(f"Failed to directly parse event files: {e}")
    
    # Sort evolution data by epoch for each parameter and heliostat
    for param_name in parameter_evolution:
        for heliostat_id in parameter_evolution[param_name]:
            parameter_evolution[param_name][heliostat_id].sort(key=lambda x: x[0])
            
    # Print summary of what was found
    print("\nParameter data extracted from Tensorboard:")
    total_data_points = 0
    for param_name, heliostat_data in parameter_evolution.items():
        param_data_points = 0
        print(f"  - {param_name}: data for {len(heliostat_data)} heliostats")
        for heliostat_id, epochs_data in heliostat_data.items():
            param_data_points += len(epochs_data)
            if len(epochs_data) > 0:
                min_val = min(v for _, v in epochs_data)
                max_val = max(v for _, v in epochs_data)
                print(f"      - {heliostat_id}: {len(epochs_data)} epochs, values range: [{min_val:.6f}, {max_val:.6f}]")
        total_data_points += param_data_points
        print(f"      Total data points for {param_name}: {param_data_points}")
    
    print(f"\nTotal parameter data points extracted: {total_data_points}")
    
    return parameter_evolution

def plot_parameter_evolution(modified_params, parameter_evolution, heliostat_ids, output_dir, log_scale=False):
    """
    Plot the evolution of parameters over training epochs, normalized to their initial error.
    
    Args:
        modified_params: Dictionary of true parameter values from CSV
        parameter_evolution: Dictionary of parameter evolution from Tensorboard
        heliostat_ids: List of heliostat IDs from the CSV
        output_dir: Directory to save the plots
        log_scale: Boolean flag to determine if y-axis should use logarithmic scale
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plot style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Process each parameter
    for param_name in parameter_evolution:
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create a color map for heliostats
        colors = plt.cm.viridis(np.linspace(0, 1, len(heliostat_ids)))
        
        # Track whether this parameter has data to plot
        has_data = False
        
        # Plot evolution for each heliostat
        for i, heliostat_id in enumerate(heliostat_ids):
            # Check if we have evolution data for this heliostat and parameter
            if heliostat_id in parameter_evolution[param_name]:
                evolution_data = parameter_evolution[param_name][heliostat_id]
                
                # Check if we have the true value for this parameter and heliostat
                if param_name in modified_params and heliostat_id in modified_params[param_name]:
                    true_value = modified_params[param_name][heliostat_id]
                    
                    # Extract epochs and values
                    epochs = [e for e, _ in evolution_data]
                    values = [v for _, v in evolution_data]
                    
                    # Get initial value (epoch 0)
                    initial_value = values[0] if values else None
                    
                    if initial_value is not None and abs(true_value - initial_value) > 1e-10:  # Avoid division by near-zero
                        # Calculate initial error
                        initial_error = initial_value - true_value
                        
                        # Normalize values based on initial error
                        normalized_values = [(v - true_value) / initial_error for v in values]
                        
                        # Plot the normalized values
                        if log_scale:
                            # For log scale, plot absolute values with small epsilon
                            epsilon = 1e-10
                            log_values = [abs(v) + epsilon for v in normalized_values]
                            plt.semilogy(epochs, log_values, 
                                    label=f"{heliostat_id}", 
                                    color=colors[i], 
                                    linewidth=2)
                        else:
                            # For linear scale, plot original normalized values
                            plt.plot(epochs, normalized_values, 
                                    label=f"{heliostat_id}", 
                                    color=colors[i], 
                                    linewidth=2)
                        
                        y_label = 'Absolute Normalized Error (Initial Error = 1)'    
                        has_data = True
                        
                    elif initial_value is not None and abs(initial_value) > 1e-10:  # Normalize on initial value
                       
                        # Normalize values based on initial value
                        normalized_values = [(v) / initial_value for v in values]
                        
                        # Plot the normalized values
                        if log_scale:
                            # For log scale, plot absolute values with small epsilon
                            epsilon = 1e-10
                            log_values = [abs(v) + epsilon for v in normalized_values]
                            plt.semilogy(epochs, log_values, 
                                    label=f"{heliostat_id}", 
                                    color=colors[i], 
                                    linewidth=2)
                        else:
                            # For linear scale, plot original normalized values
                            plt.plot(epochs, normalized_values, 
                                    label=f"{heliostat_id}", 
                                    color=colors[i], 
                                    linewidth=2)
                        
                        y_label = 'Absolute Normalized Value (Initial Value = 1)'
                        has_data = True
                        
                    elif initial_value is not None:  # Initial value is close to zero, don't normalize
                        
                        # Plot the normalized values
                        if log_scale:
                            # For log scale, plot absolute values with small epsilon
                            epsilon = 1e-10
                            log_values = [abs(v) + epsilon for v in values]
                            plt.semilogy(epochs, log_values, 
                                    label=f"{heliostat_id}", 
                                    color=colors[i], 
                                    linewidth=2)
                        else:
                            # For linear scale, plot original normalized values
                            plt.plot(epochs, values, 
                                    label=f"{heliostat_id}", 
                                    color=colors[i], 
                                    linewidth=2)
                        
                        y_label = 'Real Value'
                        has_data = True
        
        if has_data:
            # Set labels and title
            plt.xlabel("Training Epoch")
            
            if log_scale:
                plt.ylabel(y_label)
                plt.title(f"Convergence of {param_name} Parameter (Log Scale)")
            else:
                plt.ylabel(y_label)
                plt.title(f"Convergence of {param_name} Parameter")
                # Add a horizontal line at y=0 (perfect convergence)
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            # Add legend
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add grid
            plt.grid(True, which="both" if log_scale else "major", alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure
            clean_param_name = param_name.replace('/', '_').replace(' ', '_')
            scale_suffix = "log" if log_scale else "linear"
            plt.savefig(os.path.join(output_dir, f"{clean_param_name}_evolution_{scale_suffix}.png"), dpi=300)
            
            # Close the figure to free memory
            plt.close()

def analyze_parameter_convergence(csv_file, tensorboard_log_dir, output_dir="./parameter_plots", log_scale=False):
    """
    Main function to analyze parameter convergence from CSV and Tensorboard data.
    
    Args:
        csv_file: Path to CSV file with true parameter values
        tensorboard_log_dir: Directory containing Tensorboard logs
        output_dir: Directory to save parameter evolution plots
    """
    print(f"Loading parameter values from {csv_file}...")
    modified_params, heliostat_ids = load_parameters_from_csv(csv_file)
    
    print(f"Extracting parameter evolution from Tensorboard logs in {tensorboard_log_dir}...")
    parameter_evolution = extract_tensorboard_parameters(tensorboard_log_dir)
    
    print(f"Plotting parameter evolution...")
    plot_parameter_evolution(modified_params, parameter_evolution, heliostat_ids, output_dir, log_scale=log_scale)
    
    print(f"Parameter evolution plots saved to {output_dir}")
    print(f"Generated plots for {len(parameter_evolution.keys())} parameters:")
    print(f"{', '.join(parameter_evolution.keys())}")


analyze_parameter_convergence(
    csv_file="/dss/dsshome1/05/di38kid/data/simulated_data/11/parameters/modified_values.csv",
    tensorboard_log_dir="/dss/dsshome1/05/di38kid/data/results/runs/run_25041110_AM35/log",
    output_dir="/dss/dsshome1/05/di38kid/data/results/runs/run_25041110_AM35/plots",
    log_scale=False
)
