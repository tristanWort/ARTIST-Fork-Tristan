import os
import sys
import torch
import gpustat
import psutil
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 
from artist.field.kinematic_rigid_body import RigidBody

from logger import TensorboardReader

# TODO: Add names keys for parameters dict
def get_rigid_body_kinematic_parameters_from_scenario(
    kinematic: RigidBody,
) -> dict[str, torch.Tensor]:
    """
    Extract all deviation parameters and actuator parameters from a rigid body kinematic.

    Parameters
    ----------
    kinematic : RigidBody
        The kinematic from which to extract the parameters.

    Returns
    -------
    dict[str, torch.Tensor]
        The parameters from the kinematic (requires_grad is True).
    """    
    parameters_dict = torch.nn.ParameterDict()
    parameters_dict.update(kinematic.all_heliostats_position_params)
    parameters_dict.update(kinematic.all_deviations_params)
    parameters_dict.update(kinematic.all_actuators_params)
    
    return parameters_dict


def check_for_nan_grad(obj, name=None, index=None):
    """
    Recursively check for nan-gradients in Pytorch containers.
    Works with nn.ParameterDict, nn.ParameterList, nn.Parameter, etc.
    
    """
    if isinstance(obj, torch.nn.Parameter):
        if obj.grad is not None and torch.isnan(obj.grad).any():
            print(f"Paramter has nan-gradient: {name}, {index}")
            
    elif isinstance(obj, (torch.nn.ParameterDict, dict)):
        for name, param in obj.items():
            check_for_nan_grad(param, name=name, index=index)
                          
    elif isinstance(obj, (torch.nn.ParameterList, list, tuple)):
        for index, param in enumerate(obj):
            check_for_nan_grad(param, name=name, index=index)


def count_parameters(obj):
    """
    Recursively count parameters in PyTorch containers.
    Works with nn.ParameterDict, nn.ParameterList, nn.Parameter, etc.
    
    Returns:
        tuple: (num_params, num_elements) where
               num_params is the number of Parameter objects
               num_elements is the total number of values in all parameters
    """
    if isinstance(obj, torch.nn.Parameter):
        return 1, obj.numel()
    
    elif isinstance(obj, (torch.nn.ParameterDict, torch.nn.ParameterList, 
                          dict, list, tuple)):
        # Get collection of items to iterate
        if isinstance(obj, (torch.nn.ParameterDict, dict)):
            items = obj.values()
        else:  # ParameterList, list, or tuple
            items = obj
            
        # Sum up parameters in all items
        params_count = 0
        elements_count = 0
        for item in items:
            p, e = count_parameters(item)
            params_count += p
            elements_count += e
            
        return params_count, elements_count
    
    # Not a parameter or container
    return 0, 0

def create_parameter_groups(all_heliostats_params, num_heliostats):
  
    # Group by heliostat
    param_groups = []
    
    for i in range(num_heliostats):
        # 2a) Translational parameters for this heliostat
        param_groups.append({
            'params': [
                all_heliostats_params['heliostat_e'][i],
                all_heliostats_params['heliostat_n'][i],
                all_heliostats_params['heliostat_u'][i],
                all_heliostats_params['first_joint_translation_e'][i],
                all_heliostats_params['first_joint_translation_n'][i],
                all_heliostats_params['first_joint_translation_u'][i],
                all_heliostats_params['second_joint_translation_e'][i],
                all_heliostats_params['second_joint_translation_n'][i],
                all_heliostats_params['second_joint_translation_u'][i],
                # all_heliostats_params['concentrator_translation_e'][i],
                # all_heliostats_params['concentrator_translation_n'][i],
                # all_heliostats_params['concentrator_translation_u'][i]
            ],
            'lr': 5e-3,
            'name': f'heliostat_{i}_translational'
        })
        
        # 2b) Rotational parameters for this heliostat
        param_groups.append({
            'params': [
                all_heliostats_params['first_joint_tilt_n'][i],
                all_heliostats_params['first_joint_tilt_u'][i],
                all_heliostats_params['second_joint_tilt_e'][i],
                all_heliostats_params['second_joint_tilt_n'][i],
                all_heliostats_params['concentrator_tilt_e'][i],
                all_heliostats_params['concentrator_tilt_n'][i],
                all_heliostats_params['concentrator_tilt_u'][i]
            ],
            'lr': 5e-4,
            'name': f'heliostat_{i}_rotational'
        })
        
        for j in range(len(all_heliostats_params['actuators_increments'][i])):
            # 2c) Increments
            param_groups.append({
                'params': [all_heliostats_params['actuators_increments'][i][j]],
                'lr': 5e-2,
                'name': f'heliostat_{i}_actuator_{j}_increments'
            })
            
            # 2d) Stroke lengths
            param_groups.append({
                'params': [all_heliostats_params['actuators_initial_stroke_lengths'][i][j]],
                'lr': 1e-4,
                'name': f'heliostat_{i}_actuator_{j}_strokes'
            })
            
            # 2e) Offsets
            param_groups.append({
                'params': [all_heliostats_params['actuators_offsets'][i][j]],
                'lr': 1e-4,
                'name': f'heliostat_{i}_actuator_{j}_offsets'
            })
            
            # 2f) Pivot radii
            param_groups.append({
                'params': [all_heliostats_params['actuators_pivot_radii'][i][j]],
                'lr': 1e-4,
                'name': f'heliostat_{i}_actuator_{j}_radii'
            })
            
            # 2g) Initial angles
            param_groups.append({
                'params': [all_heliostats_params['actuators_initial_angles'][i][j]],
                'lr': 1e-4,
                'name': f'heliostat_{i}_actuator_{j}_angles'
            })
    
    return param_groups

def calculate_intersection(ray_origin: torch.Tensor, 
                           ray_direction: torch.Tensor, 
                           plane_center: torch.Tensor, 
                           plane_normal: torch.Tensor):
    """
    Calculate the intersection point of a batch of rays with a plane.
    
    All inputs must be given in ENU-coordinates. 
    
    Args:
        ray_origin (torch.Tensor): The point of origin of the ray (reflection point) [B, 4]
        ray_direction (torch.Tensor): The direction of the ray (normalized) [B, 4]
        plane_center (torch.Tensor): A point on the plane (center point) [B, 4]
        plane_normal (torch.Tensor): Normal vector of the plane (normalized) [B, 4]
        
    Returns:
        torch.Tensor: Intersection point in ENU-coordinates [B, 4]
        torch.Tensor: Distance from ray_origin to intersection point in meter [B, 4]
    """
    # Convert all inputs to float64 for maximum precision
    ray_origin = ray_origin.to(dtype=torch.float64)
    ray_direction = ray_direction.to(dtype=torch.float64)
    plane_center = plane_center.to(dtype=torch.float64)
    plane_normal = plane_normal.to(dtype=torch.float64)
    
    # Normalize vectors if they aren't already (using float64 precision)
    ray_direction = ray_direction / torch.norm(ray_direction, dim=-1, keepdim=True, dtype=torch.float64)
    plane_normal = plane_normal / torch.norm(plane_normal, dim=-1, keepdim=True, dtype=torch.float64)
    
    # Calculate the vector from ray origin to plane center (in float64)
    ray_to_plane = plane_center - ray_origin
    
    # Calculate the denominator (dot product of ray direction and plane normal)
    denominator = torch.sum(ray_direction * plane_normal, dim=-1, keepdim=True)
    
    # Calculate the distance along the ray to the intersection point
    # t = (ray_to_plane • plane_normal) / (ray_direction • plane_normal)
    t = torch.sum(ray_to_plane * plane_normal, dim=-1, keepdim=True) / denominator
    
    # Calculate the intersection point (maintaining float64 precision)
    intersection_point = ray_origin + t * ray_direction
    
    return intersection_point, t

def enu_point_to_target_plane(
        enu_point: torch.Tensor,
        target_center: torch.Tensor,
        plane_e: float,
        plane_u: float,
        resolution_e: int=256,
        resolution_u: int=256,
) -> list:
    """
    Map the coordinates of a point in ENU coordinates onto a target plane.

    Parameters
    ----------
    enu_point
    target_center
    plane_e
    plane_u

    Returns
    -------

    """

    d_enu = enu_point - target_center
    map_e = 0.5 + d_enu[0].item() / plane_e
    map_u = 0.5 + d_enu[2].item() / plane_u
    pixel_point = [
        resolution_e * map_e,
        resolution_u * map_u,
    ]
    return pixel_point

def percentage_near_max(image, threshold=1):
    """
    Calculate the percentage of pixels within a certain threshold of the maximum value
    in each image of the (n, w, h) tensor.


    Args:
        image (torch.Tensor):
            Input-Tensor of shape (n, w, h).
        threshold (float):
            Distance from the maximum value within which pixels are counted.

    Returns:
        numpy.ndarray oder torch.Tensor:
            An array of percentage values for each image in the batch.
    """
    if isinstance(image, torch.Tensor):
        max_vals = image.amax(dim=(1, 2), keepdim=True)  # Maximalwerte pro Bild
        mask = (image >= (max_vals - threshold))  # Maske für Pixel innerhalb des Thresholds
        percentages = mask.sum(dim=(1, 2)) / (image.shape[1] * image.shape[2]) * 100
        return percentages

    else:
        raise TypeError("Input must be a PyTorch-Tensor!")

def analyze_calibration_results(save_dir=None):
    """
    Analyze the results of a calibration run.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    log_dir : str, optional
        Directory where logs are saved
    """
    
    # Create a TensorboardReader
    reader = TensorboardReader(f'{save_dir}/log')
    
    # Get a summary of the final performance
    summary = reader.create_performance_summary()
    print("Performance Summary:")
    print(summary)
    
    plot_dir = f'{save_dir}/plots'
    os.makedirs(plot_dir, exist_ok=True)
    # Plot the loss curves
    loss_fig = reader.plot_losses(modes=['Train', 'Validation'])
    loss_fig.savefig(f'{plot_dir}/01_loss_over_epochs.png')
    
    # Plot alignment errors
    error_fig = reader.plot_metrics(metric_name='AlignmentErrors_mrad', modes=['Train', 'Validation'])
    error_fig.savefig(f'{plot_dir}/02_error_over_epochs.png')
    
    # Plot final error distribution
    # TODO: Here create plot for sun distribution
    # dist_fig = reader.visualize_final_error_distribution()
    # dist_fig.savefig(f'{plot_dir}/03_error_distribution.png')

def print_gpu_memory_usage(device, location=''):
    """Monitor GPU memory usage"""
    print("-------------------------------------------------------------------------")
    print(f"Location: {location}")
    print(f"CPU memory usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")
    mem = psutil.virtual_memory()
    total_ram = mem.total / (1024 ** 3)  # Convert bytes to GB
    print(f"Available RAM on CPU: {total_ram:.2f} GB")
    print(f"Allocated memory: {torch.cuda.memory_allocated(device=device) / 1024**2:.2f} MB")
    print(f"Reserved memory: {torch.cuda.memory_reserved(device=device) / 1024**2:.2f} MB")
    print(f"Max allocated memory: {torch.cuda.max_memory_allocated(device=device) / 1024**2:.2f} MB")
    print(f"Max reserved memory: {torch.cuda.max_memory_reserved(device=device) / 1024**2:.2f} MB")
    print(gpustat.new_query())
    print("-------------------------------------------------------------------------")
