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

def calculate_param_errors(modified_params, parameter_evolution):
    """
    Calculate errors between predicted and true values for each parameter and heliostat.
    
    Args:
        modified_params: Dictionary of true parameter values from CSV
        parameter_evolution: Dictionary of parameter evolution from Tensorboard
    
    Returns:
        param_errors: Dictionary mapping parameter names to heliostat IDs to epoch/error pairs
        max_errors: Dictionary mapping parameter names to heliostat IDs to their maximum error
    """
    param_errors = defaultdict(lambda: defaultdict(list))
    max_errors = defaultdict(lambda: defaultdict(float))
    
    # Process each parameter
    for param_name in parameter_evolution:
        for heliostat_id in parameter_evolution[param_name]:
            evolution_data = parameter_evolution[param_name][heliostat_id]
            
            # Check if we have the true value for this parameter and heliostat
            if param_name in modified_params and heliostat_id in modified_params[param_name]:
                true_value = modified_params[param_name][heliostat_id]
                
                # Calculate errors for each epoch
                errors = []
                for epoch, value in evolution_data:
                    error = value - true_value
                    errors.append((epoch, error))
                    
                # Store errors
                param_errors[param_name][heliostat_id] = errors
                
                # Calculate maximum absolute error for normalization
                if errors:
                    max_abs_error = max(abs(err) for _, err in errors)
                    max_errors[param_name][heliostat_id] = max_abs_error
            else:
                # If we don't have true values, just use the raw values
                param_errors[param_name][heliostat_id] = evolution_data
                if evolution_data:
                    max_value = max(abs(val) for _, val in evolution_data)
                    max_errors[param_name][heliostat_id] = max_value
    
    return param_errors, max_errors

def categorize_parameters(parameter_names):
    """
    Categorize parameters into translations, tilts, and actuators.
    
    Args:
        parameter_names: List of parameter names
    
    Returns:
        categories: Dictionary mapping categories to lists of parameter names
    """
    categories = {
        'translations': [],
        'tilts': [],
        'actuators': [],
        'actuators increments': [],
        'other': []
    }
    
    for param_name in parameter_names:
        # Parameters with 'translation' or 'heliostat' in their name go to translations category
        if 'translation' in param_name.lower() or 'heliostat' in param_name.lower():
            categories['translations'].append(param_name)
        elif 'tilt' in param_name.lower():
            categories['tilts'].append(param_name)
        elif 'actuator' in param_name.lower():
            if not 'increments' in param_name.lower(): 
                categories['actuators'].append(param_name)
            else:
                categories['actuators increments'].append(param_name)
        else:
            categories['other'].append(param_name)
    
    return categories

def plot_normalized_errors_by_heliostat(param_errors, max_errors, heliostat_ids, output_dir, normalize=False):
    """
    Plot normalized errors for all parameters grouped by heliostat and parameter category.
    
    Args:
        param_errors: Dictionary mapping parameter names to heliostat IDs to epoch/error pairs
        max_errors: Dictionary mapping parameter names to heliostat IDs to their maximum error
        heliostat_ids: List of heliostat IDs
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plot style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Get all parameter names
    all_params = list(param_errors.keys())
    
    # Categorize parameters
    param_categories = categorize_parameters(all_params)
    
    # Category titles for plots
    category_titles = {
        'translations': 'Translation Parameters',
        'tilts': 'Tilt Parameters',
        'actuators': 'Actuator Parameters',
        'actuators increments': 'Actuator Increments'
    }
    
    # Process each heliostat
    for heliostat_id in heliostat_ids:
        # Create plots for each category
        for category, params in param_categories.items():
            if category == 'other' or not params:
                continue  # Skip 'other' category or empty categories
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Create a color map
            colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
            
            # Track whether this heliostat has data to plot in this category
            has_data = False
            
            # Plot normalized errors for each parameter in this category
            for i, param_name in enumerate(params):
                # Check if we have error data for this parameter and heliostat
                if heliostat_id in param_errors[param_name]:
                    error_data = param_errors[param_name][heliostat_id]
                    
                    # Get maximum error for normalization
                    if normalize:
                        max_error = max_errors[param_name][heliostat_id]
                        
                        if max_error > 0:  # Avoid division by zero
                            # Extract epochs and errors
                            epochs = [e for e, _ in error_data]
                            errors = [err for _, err in error_data]
                            
                            # Normalize errors by maximum error
                            normalized_errors = [err / max_error for err in errors]
                            
                            # Plot normalized errors
                            plt.plot(epochs, normalized_errors, 
                                    label=f"{param_name}", 
                                    color=colors[i], 
                                    linewidth=2)
                    
                    else:
                        # Extract epochs and errors
                        epochs = [e for e, _ in error_data]
                        errors = [err for _, err in error_data]
                        
                        # Plot absolute errors
                        plt.plot(epochs, errors, 
                                label=f"{param_name}", 
                                color=colors[i], 
                                linewidth=2)
                    
                    has_data = True
            
            if has_data:
                # Set labels and title
                plt.xlabel("Training Epoch")
                if normalize: 
                    plt.ylabel("Normalized Parameter Error (Error in Prediction / Max Error)")
                else:
                    plt.ylabel("Parameter Error (Prediction - True Value)")
                plt.title(f"{category_titles[category]} - Heliostat {heliostat_id}")
                
                # Add horizontal line at y=0 (perfect convergence)
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                
                # Add legend (with smaller font to fit all parameters)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                
                # Add grid
                plt.grid(True, alpha=0.3)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save the figure
                clean_category = category.replace('/', '_').replace(' ', '_')
                plt.savefig(os.path.join(output_dir, f"heliostat_{heliostat_id}_{clean_category}.png"), dpi=300)
                
                # Close the figure to free memory
                plt.close()

def analyze_parameter_convergence(csv_file, tensorboard_log_dir, output_dir="./parameter_plots"):
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
    
    print(f"Calculating parameter errors...")
    param_errors, max_errors = calculate_param_errors(modified_params, parameter_evolution)
    
    print(f"Plotting normalized errors by heliostat...")
    plot_normalized_errors_by_heliostat(param_errors, max_errors, heliostat_ids, output_dir)
    
    print(f"Parameter error plots saved to {output_dir}")
    print(f"Generated plots for {len(heliostat_ids)} heliostats across 3 parameter categories")
    
    # Print count of parameters in each category
    param_categories = categorize_parameters(list(parameter_evolution.keys()))
    for category, params in param_categories.items():
        if category != 'other':
            print(f"{category.capitalize()}: {len(params)} parameters")


analyze_parameter_convergence(
    csv_file="/dss/dsshome1/05/di38kid/data/results/runs/run_2504151605_AM35/parameters/original_values.csv",
    tensorboard_log_dir="/dss/dsshome1/05/di38kid/data/results/runs/run_2504151605_AM35/log",
    output_dir="/dss/dsshome1/05/di38kid/data/results/runs/run_2504151605_AM35/plots"
)
