import torch
import torch.nn.functional as F
import os
import sys
import pathlib
import numpy as np

from matplotlib import pyplot as plt
from typing import Union

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 

from artist.raytracing.heliostat_tracing import HeliostatRayTracer


def align_and_raytrace(scenario,
                       incident_ray_directions: torch.tensor,
                       target_areas: list(),
                       aim_points=None,
                       align_with_motor_positions=False,
                       motor_positions=None,
                       seed=42,
                       device='cuda'):
        """
        Perform aiming and raytracing for the heliostat field for the given scenario.
        
        For alignment there are two options.
            1) Align with incident ray directions and aim points.
            2) Align with motor positions.
        """
        # Auxiliary referencing to kinematic
        kinematic = scenario.heliostat_field.rigid_body_kinematic
        
        if not align_with_motor_positions:
            # Set aim points and then align with incident ray directions
            kinematic.aim_points = aim_points
            # Align with incident ray directions
            scenario.heliostat_field.align_surfaces_with_incident_ray_direction(
                incident_ray_direction=incident_ray_directions,
                round_motor_pos=True,
                device=device
                )
        
        else:
            # Align with motor positions
            scenario.heliostat_field.align_surfaces_with_motor_positions(
               motor_positions=motor_positions,
               device=device
               )
        
        # Initiate Raytracer
        raytracer = HeliostatRayTracer(scenario=scenario, 
                                       world_size=1, 
                                       rank=0, 
                                       batch_size=1, 
                                       random_seed=seed) 
        # Raytrace and store bitmaps
        final_bitmaps = raytracer.trace_rays_separate(incident_ray_directions=incident_ray_directions,
                                                      target_areas=target_areas,
                                                      device=device)
        return final_bitmaps
    
    
def apply_artificial_blocking(image: torch.Tensor, block_ratio: float = 0.5, threshold: float = 0.1, sigma: float = 2.0):
    """
    Applies artificial blocking to the lower portion of a heliostat focal spot image.

    Parameters
    ----------
    image : torch.Tensor
        2D tensor of shape (H, W), values should be normalized [0, 1].
    block_ratio : float
        Fraction (0 to 1) of the focal spot height to be shaded from the bottom up.
    threshold : float
        Intensity threshold to define focal spot presence.
    sigma : float
        Standard deviation of the Gaussian noise added to soften the shading edge.

    Returns
    -------
    torch.Tensor
        The modified image with artificial blocking applied.
    """
    assert image.dim() == 2, "Input image must be a 2D tensor (H, W)."
    H, W = image.shape

    # Find bounding box of the focal spot based on threshold
    mask = image > threshold
    row_indices = torch.any(mask, dim=1).nonzero().squeeze()
    if row_indices.numel() == 0:
        return image  # No focal spot found, return original image
    
    top_idx = row_indices[0].item()
    bottom_idx = row_indices[-1].item()
    focal_height = bottom_idx - top_idx + 1
    
    # Determine blocking start based on ratio
    block_start = int(bottom_idx - block_ratio * focal_height)
    
    # Generate vertical Gaussian mask
    vertical_mask = torch.zeros(H, 1, device=image.device)
    edge = torch.arange(H, device=image.device).unsqueeze(1)
    gaussian_edge = torch.sigmoid(-(edge - block_start) / sigma)

    # Expand to full image width
    gaussian_mask = gaussian_edge.expand(-1, W)

    # Apply soft blocking
    shaded_image = image * gaussian_mask
    return shaded_image


def create_gradient_kernel(size=5, direction='down', randomize=True, random_strength=0.3, device: Union[str, torch.device] = 'cpu'):
    """
    Create a gradient detection kernel for simulating heliostat blocking in PyTorch.

    Parameters:
    - size: Size of the kernel (odd number)
    - direction: Direction of gradient ('up', 'down', 'left', 'right')
    - randomize: Whether to add randomness to the kernel
    - random_strength: Strength of randomization (0.0 to 1.0)
    - device: PyTorch device ('cpu' or 'cuda')

    Returns:
    - kernel: 2D tensor representing the gradient kernel
    """
    # Ensure size is odd
    if size % 2 == 0:
        size += 1

    # Create base kernel
    kernel = torch.zeros((size, size), device=device)

    if direction == 'up':
        # Top half is negative, bottom half is positive
        kernel[:size // 2, :] = -1
        kernel[size // 2 + 1:, :] = 1
    elif direction == 'down':
        # Bottom half is negative, top half is positive
        kernel[:size // 2, :] = 1
        kernel[size // 2 + 1:, :] = -1
    elif direction == 'left':
        # Left half is negative, right half is positive
        kernel[:, :size // 2] = -1
        kernel[:, size // 2 + 1:] = 1
    elif direction == 'right':
        # Right half is negative, left half is positive
        kernel[:, :size // 2] = 1
        kernel[:, size // 2 + 1:] = -1
    else:
        raise ValueError("direction must be 'up', 'down', 'left', or 'right'")

    # Add emphasis to edge rows/columns for stronger gradient detection
    if direction in ['up', 'down']:
        kernel[0, :] *= 2  # Emphasize top row
        kernel[-1, :] *= 2  # Emphasize bottom row
    else:
        kernel[:, 0] *= 2  # Emphasize leftmost column
        kernel[:, -1] *= 2  # Emphasize rightmost column

    # Normalize kernel
    kernel = kernel / torch.abs(kernel).sum()

    # Add randomness if requested
    if randomize:
        random_factor = 1 + (2 * random_strength) * (torch.rand_like(kernel) - 0.5)
        kernel = kernel * random_factor
        # Re-normalize
        kernel = kernel / torch.abs(kernel).sum()

    # Reshape kernel for PyTorch's conv2d: [out_channels, in_channels, height, width]
    kernel = kernel.reshape(1, 1, size, size)

    return kernel * 180


def apply_blocking_convolution(flux_map, kernel_size=5, block_strength=0.5,
                               direction='up', randomize=True, random_strength=0.3, device: Union[str, torch.device] = 'cpu'):
    """
    Apply blocking effect using convolution in PyTorch.

    Parameters:
    - flux_map: 2D tensor representing the flux distribution
    - kernel_size: Size of the gradient kernel
    - block_strength: Strength of the blocking effect (0.0 to 1.0)
    - direction: Direction of gradient detection
    - randomize: Whether to add randomness
    - random_strength: Strength of randomization
    - device: PyTorch device ('cpu' or 'cuda')

    Returns:
    - blocked_map: 2D tensor with blocking effects applied
    - gradient: Gradient map showing detected gradients
    - kernel: The kernel used for convolution
    """
    # Move flux_map to specified device if it's not already there
    if flux_map.device != device:
        flux_map = flux_map.to(device)

    # If flux_map is a 2D tensor, add batch and channel dimensions
    while flux_map.dim() < 4:
        flux_map = flux_map.unsqueeze(0).unsqueeze(0)  # [batch, channel, height, width]

    # Create the gradient kernel
    kernel = create_gradient_kernel(kernel_size, direction, randomize, random_strength, device)

    # Calculate padding needed to maintain input size
    pad = kernel_size // 2

    # Apply convolution to detect gradients
    gradient = F.conv2d(flux_map, kernel, padding=pad)

    # Only consider positive gradients (for blocking effect)
    positive_gradient = torch.clamp(gradient, min=0)

    # Apply blocking effect based on detected gradients
    blocked_map = flux_map * (1 - block_strength * positive_gradient)

    # Ensure values stay within valid range
    blocked_map = torch.clamp(blocked_map, min=0, max=flux_map.max())

    # Remove batch and channel dimensions if input was 2D
    if blocked_map.size(0) == 1 and blocked_map.size(1) == 1:
        blocked_map = blocked_map.squeeze(0).squeeze(0)
        gradient = gradient.squeeze(0).squeeze(0)

    return blocked_map, gradient, kernel.squeeze(0).squeeze(0)  # Return kernel in 2D format


def aim_and_shoot_and_save_bitmaps(scenario,
                                   name: str,
                                   incident_ray_directions: torch.tensor,
                                   target_areas: list(),
                                   aim_points=None,
                                   align_with_motor_positions=False,
                                   motor_positions=None,
                                   seed=42,
                                   device='cuda'):
        """
        Perform aiming and raytracing for the heliostat field for the given scenario.
        
        For alignment there are two options.
            1) Align with incident ray directions and aim points.
            2) Align with motor positions.
        """
        # Auxiliary referencing to kinematic
        kinematic = scenario.heliostat_field.rigid_body_kinematic
        
        if not align_with_motor_positions:
            # Set aim points and then align with incident ray directions
            kinematic.aim_points = aim_points
            # Align with incident ray directions
            scenario.heliostat_field.align_surfaces_with_incident_ray_direction(
                incident_ray_direction=incident_ray_directions,
                round_motor_pos=True,
                device=device
                )
        
        else:
            # Align with motor positions
            print(f'Align with motor positions: {motor_positions}')
            scenario.heliostat_field.align_surfaces_with_motor_positions(
               motor_positions=motor_positions,
               device=device
               )
        
        # Initiate Raytracer
        raytracer = HeliostatRayTracer(scenario=scenario, 
                                       world_size=1, 
                                       rank=0, 
                                       batch_size=1, 
                                       random_seed=seed) 
        # Raytrace and store bitmaps
        final_bitmaps = raytracer.trace_rays_separate(incident_ray_directions=incident_ray_directions,
                                                      target_areas=target_areas,
                                                      device=device)
        
        # Save bitmaps in one img file
        fig, axs = plt.subplots(nrows=final_bitmaps.shape[0], figsize=(10, 10))
        for i in range(final_bitmaps.shape[0]):
            print(f"\tPixel sum in flux image {i+1}:", final_bitmaps[i].sum().item())
            axs[i].imshow(final_bitmaps[i].cpu().detach(), cmap="inferno")

        
        fig.suptitle(f"Flux Density Distributions: {name}")
        save_dir = pathlib.Path('/dss/dsshome1/05/di38kid/data/results/simulated_data/01/raytracing')
        save_path = save_dir / f'{name}_raytracing_sep.png'
        plt.tight_layout()
        plt.savefig(save_path)
        print("Bitmpas were saved!")
