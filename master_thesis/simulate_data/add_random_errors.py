import torch
import numpy as np
import pandas as pd
import copy
import os
import sys

from typing import Union

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 
from artist.field.kinematic_rigid_body import RigidBody


def save_parameters_to_csv(parameters: dict(), save_dir: str, file_name: str, heliostat_names: None) -> pd.DataFrame:
    # Add heliostat column headers
    if heliostat_names is None:
        num_heliostats = len(next(iter(parameters.values())))
        heliostat_cols = [f"heliostat_{index}" for index in range(num_heliostats)]
    else:
        heliostat_cols = [f"{heliostat}" for heliostat in heliostat_names]
    
    # Convert dictionary to Dataframe
    param_df = pd.DataFrame(parameters)
    param_df_t = param_df.transpose()
    param_df_t.columns = heliostat_cols
    param_df_t.index.name = "Parameter"
    
    # Save Dataframe
    param_df_t.to_csv(os.path.join(save_dir, file_name))


def add_error_terms(parameters: torch.nn.ParameterList, error_config: dict(), seed=42, device: Union[torch.device, str] = "cuda"):
    """
    Add random errors to the parameters by looking up the errors magnituted in 
    the provided config dictionary.
    """   
    # Set device
    device = torch.device(device)
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Save errors in dictionary
    original_dict = {}
    modified_dict = {}
    error_dict = {}
    
    # Iterate over the items in the config dictionary to calculate param offsets    
    for param_name, error_mag in error_config.items():
        param_list = parameters[param_name]           
        
        originals = []
        modified = []
        errors = []
        
        nested = False
        
        for i, param in enumerate(param_list):
            
            # Needed for actuator params: There are two actuators per heliostat
            if isinstance(param, (torch.nn.ParameterList, list, tuple)):
                
                nested = True
                originals.append([])
                modified.append([])
                errors.append([])
                
                for j, p in enumerate(param):
                    # Generate random error within specified magnitude
                    error = (torch.rand(1) * 2 - 1) * error_mag 
                    
                    # Store original parameter value for reference
                    original_value = p.detach().clone()
                    originals[-1].append(original_value.item())
                    
                    # Add error to parameter
                    p.data = p.data + error.item()
                    modified[-1].append(p.data.item())
                    
                    # Store errors in nested list
                    errors[-1].append(error.item())
                    
            else:
                    
                # Generate random error within specified magnitude
                error = (torch.rand(1) * 2 - 1) * error_mag
                
                # Store original parameter value for reference
                original_value = param.detach().clone()
                originals.append(original_value.item())
                
                # Add error to parameter
                param.data = param.data + error.item()
                modified.append(param.data.item())
                
                # Record the error for validation
                errors.append(error.item())
        
        # If there is more than one param per heliostat, expand dictionaries
        if nested:
            for j in range(len(originals[0])):
                original_dict[f"{param_name}_{j+1}"] = [hel[j] for hel in originals]
                modified_dict[f"{param_name}_{j+1}"] = [hel[j] for hel in modified]
                error_dict[f"{param_name}_{j+1}"] = [hel[j] for hel in errors]
        
        else:
            
            original_dict[param_name] = originals
            modified_dict[param_name] = modified
            error_dict[param_name] = errors
    
    return original_dict, modified_dict, error_dict
    

# TODO: Make a class out of this
def add_random_errors_to_kinematic(kinematic: RigidBody, save_dir: str, heliostat_names=None, seed=42, device="cuda"):
    """
    Add random offsets to heliostat kinematic parameters for training data generation.
    
    Args:
        kinematic: The original kinematic object with parameter dictionaries
        heliostat_name: List of heliostat names
        save_dir: Directory path, where Dataframes will be saved to
        seed: Random seed for reproducibility
        device: Device where all torch objects will be stored on
        
    Returns:
        modified_kinematic: A copy of the original kinematic with added errors
        error_values: Dictionary containing the actual error values added (for validation)
    """
    # Set device
    device = torch.device(device)
    
    # Make sure that the number of heliostats matches the size of paramters in the kinematic instance
    assert len(heliostat_names) == len(next(iter(kinematic.all_heliostats_position_params.values()))), "\
    Number of heliostat names does not match the size of found parameters in the kinematic instance."
    
    # Data frames to store parameters and errors
    original_values_dict = {}
    modified_values_dict = {}
    error_values_dict = {}
    
    # Create a deep copy of the original kinematic to avoid modifying it
    modified_kinematic = copy.deepcopy(kinematic)
    modified_kinematic.to(device)
    
    modified_params = {
        'all_heliostats_position_params': modified_kinematic.all_heliostats_position_params,
        'all_deviations_params': modified_kinematic.all_deviations_params,
        'all_actuators_params': modified_kinematic.all_actuators_params
    }    
    
    # Dictionary to store the error values for later validation
    error_values = {
        'all_heliostats_position_params': {},
        'all_deviations_params': {},
        'all_actuators_params': {},
    }
    
    # Define error magnitudes for different parameter types
    error_config = {
        'all_heliostats_position_params': {
            'heliostat_e': 0.1,  # meters
            'heliostat_n': 0.1,  # meters
            'heliostat_u': 0.1,  # meters
        },
        'all_deviations_params': {
            # Translations (meters)
            'first_joint_translation_e': 0.05,
            'first_joint_translation_n': 0.05,
            'first_joint_translation_u': 0.01,
            'second_joint_translation_e': 0.01,
            'second_joint_translation_n': 0.01,
            'second_joint_translation_u': 0.005,
            'concentrator_translation_e': 0.0,
            'concentrator_translation_n': 0.0,
            'concentrator_translation_u': 0.0,
            
            # Tilts (radians)
            'first_joint_tilt_n': 0.01,
            'first_joint_tilt_u': 0.01,
            'second_joint_tilt_e': 0.005,
            'second_joint_tilt_n': 0.005,
            'concentrator_tilt_e': 0.0,
            'concentrator_tilt_n': 0.0,
            'concentrator_tilt_u': 0.0,
        },
        'all_actuators_params': {
            'actuators_increments': 300,  # steps / m
            'actuators_initial_stroke_lengths': 0.01,
            'actuators_offsets': 0.01,
            'actuators_pivot_radii': 0.001,
            'actuators_initial_angles': 0.003,  # radians
        }
    }
    
    # Calculate errors per group, add them to the parameters and store them for later
    for group, error_group_config in error_config.items():
        group_originals, group_modified, group_errors = add_error_terms(modified_params[group], error_group_config, seed, device)
        original_values_dict.update(group_originals)
        modified_values_dict.update(group_modified)
        error_values_dict.update(group_errors)  
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    save_parameters_to_csv(original_values_dict, save_dir, 'original_values.csv', heliostat_names)
    save_parameters_to_csv(modified_values_dict, save_dir, 'modified_values.csv', heliostat_names)
    save_parameters_to_csv(error_values_dict, save_dir, 'error_values.csv', heliostat_names)
    
    return modified_kinematic
    
