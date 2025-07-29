import torch
# import gpustat
# import psutil
from typing import Union, Tuple, List, Dict, Optional

import torch.nn.functional as F

# Add local artist path for raytracing with multiple parallel heliostats.
import os
import sys
artist_repo = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, artist_repo)  
from artist.field.kinematic_rigid_body import RigidBody


def count_parameters(obj: Union[torch.nn.Parameter, 
                                torch.nn.ParameterList, 
                                torch.nn.ParameterDict,
                                Dict, List, Tuple]):
    """
    Recursively count parameters in PyTorch containers.
    Works with nn.ParameterDict, nn.ParameterList, nn.Parameter, etc.
    
    Returns
    -------
    (int, int)
        The number of counted parameters and counted elements.
    """
    if isinstance(obj, torch.nn.Parameter):
        return 1, obj.numel()
    
    elif isinstance(obj, (torch.nn.ParameterDict, torch.nn.ParameterList, 
                          Dict, List, Tuple)):
        # Get collection of items to iterate
        if isinstance(obj, (torch.nn.ParameterDict, Dict)):
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


# TODO: Add names keys for parameters dict
def get_rigid_body_kinematic_parameters_from_scenario(kinematic: RigidBody):
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
    
    parameter_count, element_count = count_parameters(parameters_dict)
    
    return parameters_dict, parameter_count, element_count


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
    
    # Normalize vectors if they aren't already
    ray_direction = ray_direction / torch.norm(ray_direction, dim=-1, keepdim=True)
    plane_normal = plane_normal / torch.norm(plane_normal, dim=-1, keepdim=True)
    
    # Calculate the vector from ray origin to plane center
    ray_to_plane = plane_center - ray_origin
    
    # Calculate the denominator (dot product of ray direction and plane normal)
    denominator = torch.sum(ray_direction * plane_normal, dim=-1, keepdim=True)
    
    # Calculate the distance along the ray to the intersection point
    # t = (ray_to_plane • plane_normal) / (ray_direction • plane_normal)
    t = torch.sum(ray_to_plane * plane_normal, dim=-1, keepdim=True) / denominator
    
    # Calculate the intersection point
    intersection_point = ray_origin + t * ray_direction
    
    return intersection_point, t


def check_for_nan_grad(obj: Union[torch.nn.Parameter, 
                                torch.nn.ParameterList, 
                                torch.nn.ParameterDict,
                                Dict, List, Tuple], 
                       name=None, 
                       index=None):
    """
    Recursively check for nan-gradients in Pytorch containers.
    """
    if isinstance(obj, torch.nn.Parameter):
        if obj.grad is not None and torch.isnan(obj.grad).any():
            print(f"parameter {name}", "is nan")
            return True
        return False
            
    elif isinstance(obj, (torch.nn.ParameterDict, Dict)):
        for name, param in obj.items():
            if check_for_nan_grad(param, name=name, index=index):
                return True
        return False
                          
    elif isinstance(obj, (torch.nn.ParameterList, List, Tuple)):
        for index, param in enumerate(obj):
            if check_for_nan_grad(param, name=name, index=index):
               return True 
        return False
        
    return False
    

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


def find_soft_contour_vertical(bitmaps: torch.Tensor, threshold: float = 0.5, sharpness: float = 20.0):
    """
    Differentiable vertical contour extraction with soft binarization and erosion.
    
    Parameters:
    - tensor_img: Tensor with shape [B, 1, H, W], [1, H, W], or [H, W], values ∈ [0, 1]
    - threshold: Soft threshold center for binarization
    - sharpness: Controls steepness of sigmoid (higher = sharper threshold)

    Returns:
    - Soft contour mask of shape [B, 1, H, W], values ∈ [0, 1]
    """
    original_shape = bitmaps.shape  # expect [B, Hel, H, W]
    # Normalize input to 4D [B, 1, H, W]
    if bitmaps.dim() != 4:
        if bitmaps.dim() == 2:  # [H, W]
            bitmaps = bitmaps.unsqueeze(0).unsqueeze(0)
        elif bitmaps.dim() == 3:  # [1, H, W] or [B, H, W]
            if bitmaps.shape[0] == 1:
                bitmaps = bitmaps.unsqueeze(0)  # [1, 1, H, W]
            else:
                bitmaps = bitmaps.unsqueeze(1)  # [B, 1, H, W]
        else:
            raise ValueError(f"Number of dimensions for input bitmaps not excepted: {bitmaps.dim()}")

    # Reshape and add 1 channel dimension
    bitmaps = bitmaps.reshape(-1, 1, original_shape[2], original_shape[3])  # [B * Hel, 1, H, W]
    
    bitmaps = bitmaps.float()
    B, C, H, W = bitmaps.shape
    device = bitmaps.device

    # Sobel kernel for vertical edges
    sobel_kernel = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    # 1. Edge detection (vertical gradient)
    padded = F.pad(
        (bitmaps - threshold) * sharpness, (1, 1, 1, 1), mode='replicate'
    )
    grad_out = F.conv2d(padded, sobel_kernel)  # [B, 1, H, W]
    edge_mask = torch.sigmoid(grad_out)  # soft vertical edge response

    # 2. Soft thresholding
    binary = torch.sigmoid((bitmaps - threshold) * sharpness)  # [B, 1, H, W]

    # 3. Soft erosion (via 3x3 average filter)
    erosion_kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0
    padded_binary = F.pad(binary, (1, 1, 1, 1), mode='replicate')
    neighborhood_mean = F.conv2d(padded_binary, erosion_kernel)  # [B, 1, H, W]

    # 4. Contour strength = difference between center and neighborhood
    contour_strength = binary - neighborhood_mean
    soft_contour = (torch.sigmoid(contour_strength * sharpness) - 0.5) * 2
    soft_contour = soft_contour.clamp(0, 1)

    # Combine with vertical edge mask to emphasize vertical features
    output = soft_contour * edge_mask
    output = output.reshape(original_shape)
    return output


def normalize_images(images: torch.Tensor, eps: float = 1e-8):
    """
    Normalize all images (B, H, W) to [0, 1] which are above the threshold value.
    Images that are below the threshold value will be set to zero pixels.
    """
    if images.dim() == 2:
        images = images.unsqueeze(0)
    B = images.shape[0]
    images_flat = images.view(B, -1)
    min_vals = images_flat.min(dim=1, keepdim=True).values.view(B, 1, 1)
    max_vals = images_flat.max(dim=1, keepdim=True).values.view(B, 1, 1)
    return (images - min_vals) / (max_vals - min_vals + eps)


def normalize_images_with_threshold(images: torch.Tensor, threshold: float, eps: float = 1e-8):
    
    B = images.shape[0]
    images_flat = images.view(B, -1)
    
    # Compute the sum of pixels per image
    pixel_sums = images_flat.sum(dim=1)
    
    # Create a mask for images where sum > threshold
    mask = pixel_sums > threshold
    
    # Prepare empty tensor for output
    output = torch.zeros_like(images)
    
    if mask.any():
        # Normalize only the images where mask is True
        images_to_normalize = images[mask]
        images_flat = images_to_normalize.view(images_to_normalize.shape[0], -1)
        
        min_vals = images_flat.min(dim=1, keepdim=True).values.view(-1, 1, 1)
        max_vals = images_flat.max(dim=1, keepdim=True).values.view(-1, 1, 1)
        
        normalized = (images_to_normalize - min_vals) / (max_vals - min_vals + eps)
        
        # Insert normalized images back into output tensor
        output[mask] = normalized
    
    return output


def normalize_and_interpolate(bitmaps: torch.Tensor, num_interpolations=0, scale_factor=2, target_size=256):
    """
    
    """
    while bitmaps.dim() < 4:  # to [B, H_idx, W, H]
        bitmaps = bitmaps.unsqueeze(0)

    min_vals = bitmaps.amin(dim=(2, 3), keepdim=True)
    max_vals = bitmaps.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)  # avoid division by 0
    normalized = (bitmaps - min_vals) / max_vals

    if isinstance(target_size, int):
        target_size = (target_size, target_size)
        
    for _ in range(num_interpolations):
        normalized = F.interpolate(normalized, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        normalized = F.interpolate(normalized, size=target_size, mode='bilinear', align_corners=False)

    return normalized  # [B, H_idx, W, H]


def enu_point_to_target_plane(
        enu_point: torch.Tensor,
        target_center: torch.Tensor,
        plane_e: float,
        plane_u: float,
        resolution_e=256,
        resolution_u=256,
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


def get_bitmap_indices_from_center_coordinates(
    size: (int, int),
    center_coordinates: torch.Tensor,
    target_center: torch.Tensor,
    plane_e: float,
    plane_u: float,
    round=True,
    device: Union[torch.device, str] = "cuda",
):
    """
    Calculate the bitmap indices corresponding to a center of mass in ENU coordinates.

    Parameters
    ----------
    size : ()
        Width and Height of pixels
    center_coordinates : torch.Tensor
        The ENU coordinates of the center of mass.
    target_center : torch.Tensor
        The position of the center of the target.
    plane_e : float
        The width of the target surface.
    plane_u : float
        The height of the target surface.
    device : Union[torch.device, str]
        The device on which to initialize tensors (default is cuda).

    Returns
    -------
    Tuple[int, int]
        The (u_idx, e_idx) bitmap indices corresponding to the center.
    """
    device = torch.device(device)
    height, width = size

    # Basis vectors defining the target surface
    de = torch.tensor([-plane_e, 0.0, 0.0, 0.0], device=device)
    du = torch.tensor([0.0, 0.0, plane_u, 0.0], device=device)

    # Convert from global center_coordinates to local normalized (0 to 1) coordinates
    center_coordinates  = center_coordinates.to(dtype=torch.float32)
    local_vector = center_coordinates - (target_center - 0.5 * (de + du))

    # Projection onto target plane basis vectors
    e_ratio = torch.dot(local_vector, de) / torch.dot(de, de)
    u_ratio = torch.dot(local_vector, du) / torch.dot(du, du)
    
    def safe_nan_handling(ratio: torch.Tensor, device: torch.device):
        if torch.isnan(ratio):
            return torch.tensor(0.5, device=device)
        else:
            return ratio
    
    # Make sure that ratios are not nan
    e_ratio = safe_nan_handling(e_ratio, device)
    u_ratio = safe_nan_handling(u_ratio, device)   
    
    # Convert to bitmap indices (float index relative to pixel center)
    e_idx = (e_ratio * width).clamp(0, width - 1)
    u_idx = ((1 - u_ratio) * height).clamp(0, height - 1)

    if round:
        return int(u_idx.round().item()), int(e_idx.round().item())
    return u_idx.item(), e_idx.item()


def print_gpu_memory_usage(device, location=''):
    """Monitor GPU memory usage"""
    print("-------------------------------------------------------------------------")
    print(f"Location: {location}")
    # print(f"CPU memory usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")
    # mem = psutil.virtual_memory()
    total_ram = mem.total / (1024 ** 3)  # Convert bytes to GB
    print(f"Available RAM on CPU: {total_ram:.2f} GB")
    print(f"Allocated memory: {torch.cuda.memory_allocated(device=device) / 1024**2:.2f} MB")
    print(f"Reserved memory: {torch.cuda.memory_reserved(device=device) / 1024**2:.2f} MB")
    print(f"Max allocated memory: {torch.cuda.max_memory_allocated(device=device) / 1024**2:.2f} MB")
    print(f"Max reserved memory: {torch.cuda.max_memory_reserved(device=device) / 1024**2:.2f} MB")
    # print(gpustat.new_query())
    print("-------------------------------------------------------------------------")
