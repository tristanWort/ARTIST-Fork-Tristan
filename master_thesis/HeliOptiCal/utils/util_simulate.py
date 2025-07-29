import torch
import torch.nn.functional as F
import os
import sys
import pathlib
import numpy as np
import math

from matplotlib import pyplot as plt
from typing import Union
from pathlib import Path

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
                       resolution_eu=(256, 256),
                       raytracer=None,
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
                round_motor_pos=False,
                device=device
                )
            
        else:
            # Align with motor positions
            scenario.heliostat_field.align_surfaces_with_motor_positions(
            motor_positions=motor_positions,
            device=device
            )
        
        # Initiate Raytracer
        if raytracer is None:
            raytracer = HeliostatRayTracer(scenario=scenario, 
                                        world_size=1, 
                                        rank=0, 
                                        batch_size=1, 
                                        random_seed=seed,
                                        bitmap_resolution_e=resolution_eu[0],
                                        bitmap_resolution_u=resolution_eu[1]) 
        # Raytrace and store bitmaps
        bitmaps = raytracer.trace_rays_separate(incident_ray_directions=incident_ray_directions,
                                                target_areas=target_areas,
                                                device=device)
        for i in range(bitmaps.shape[0]):
            bitmaps[i] = gaussian_filter_2d(bitmaps[i], sigma=2.5)
        # bitmaps = normalize_and_interpolate(bitmaps, num_interpolations=8, target_size=960).squeeze(0)
        return bitmaps
    

def plot_center_cross(center_indices: (int, int), cross_size: int = 10, cross_color: str = "red", cross_linewidth: float = 3):
    u_idx, e_idx = center_indices
    # Draw horizontal and vertical lines to form a cross
    plt.plot(
        [e_idx - cross_size, e_idx + cross_size],
        [u_idx, u_idx],
        color=cross_color,
        linewidth=cross_linewidth,
    )
    plt.plot(
        [e_idx, e_idx],
        [u_idx - cross_size, u_idx + cross_size],
        color=cross_color,
        linewidth=cross_linewidth,
    )
         

def save_bitmap(bitmap: torch.Tensor, save_under: Path):
    width_in_pixels, height_in_pixels = 512, 512
    dpi = 100  # Can be any value
    width_in_inches = width_in_pixels / dpi
    height_in_inches = height_in_pixels / dpi

    plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
    plt.imshow(bitmap.cpu().detach(), cmap="gray")
    plt.axis("off")  # Hides both x and y axes
    plt.savefig(save_under, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()
         

def save_bitmap_with_center_cross(bitmap: torch.Tensor, center_indices: (int, int), save_under: Path):
    """
    Save a bitmap image with a red cross marking the predicted center.

    Parameters
    ----------
    bitmap : torch.Tensor
        The bitmap image to be saved (2D tensor).
    center_indices : Tuple[int, int]
        The (u_idx, e_idx) center of the flux bitmap to mark with a cross.
    save_under : Path
        The path to save the image file.
    """
    width_in_pixels, height_in_pixels = 512, 512
    dpi = 100
    width_in_inches = width_in_pixels / dpi
    height_in_inches = height_in_pixels / dpi

    plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
    plt.imshow(bitmap.cpu().detach(), cmap="gray")
    
    plot_center_cross(center_indices)
    plt.axis("off")
    plt.savefig(save_under, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()
    

def estimate_blocking_factor(sun_elevation: torch.Tensor, heliostat_height: torch.Tensor, heliostat_spacing: torch.Tensor, translation_to_target: torch.Tensor):
    """
    Function returns a conservative approximation of the relative factor [0, 1] of a heliostat's surface, which is blocked by
    the front row of heliostats at a given sun position and for a given heliostat. Uses simplified geometric relationships.
    
    Parameters
    ----------
    sun_elevation : torch.Tensor
        Elevation angle of the sun position in degrees [°].
    heliostat_height : torch.Tensor
        Vertical dimension of the heliostat mirror (sum over facets) in meters.
    helisotat_spacing : torch.Tensor
        Spacing along north-south direction to the front heliostat row in meters.
    translation_to_target : torch.Tensor
        Translation in ENU coordinates from heliostat center to target center.  
    """ 
    device = sun_elevation.device
    assert 90 >= sun_elevation.item() >= 0, f"sun_elevation should be [0°, 90°], not: {sun_elevation.item()}."
    assert heliostat_height.item() > 0, f"heliostat_height should be positive, not: {heliostat_height.item()}."
    assert heliostat_spacing.item() > 0, f"heliostat_spacing should be positive, not {heliostat_spacing.item()}."
    assert translation_to_target.shape[0] >= 3, f"Expected translation_to_target to be given in ENU-coordinates, but given dimensions: {translation_to_target.dim()}."
    
    # 1) Compute elevation angle for target reflection direction, i.e. heliostat -> target
    e_translation, n_translation, u_translation = translation_to_target[0], translation_to_target[1], translation_to_target[2]
    horizontal_norm = torch.sqrt(e_translation**2 + n_translation**2)
    local_target_elevation = torch.atan2(u_translation, horizontal_norm)
    
    # 2) Compute the elevation angle of the heliostat surface normal
    sun_elevation = torch.deg2rad(sun_elevation)
    heliostat_normal_elevation = (local_target_elevation + sun_elevation) / 2
    
    # 3) Compute the elevation angle of the heliostat surface
    heliostat_surface_elevation = torch.deg2rad(torch.tensor(90.0, device=device)) - heliostat_normal_elevation 
    
    # 4) Compute factor of blocked surface area relative to total surface area
    step_1 = torch.tan(local_target_elevation)
    step_2 = heliostat_spacing / heliostat_height
    step_3 = step_1 / torch.cos(heliostat_surface_elevation)
    step_4 = torch.tan(heliostat_surface_elevation) + step_1
    f_blocked = 1.0 - step_2 * step_3 / step_4
    
    # 5) Compute geometric condition for blocking
    step_5 = heliostat_height * torch.sin(heliostat_surface_elevation)
    step_6 = heliostat_spacing * step_1
    step_7 = step_1 / (torch.tan(heliostat_surface_elevation) + step_1)
    check_block = step_6 / step_5 * (1.0 -  step_7)
    if 0 <= check_block.item() <= 1:
        return torch.clamp(f_blocked, min=0.0, max=1.0).item()
    
    return 0.0
    

def estimate_shading_factor(sun_elevation: torch.Tensor, heliostat_height: torch.Tensor, heliostat_spacing: torch.Tensor, translation_to_target: torch.Tensor):
    """
    Function returns a conservative approximation of the relative factor [0, 1] of a heliostat's surface, which is shaded by
    the front row of heliostats at a given sun position and for a given heliostat. Uses simplified geometric relationships.
    
    Parameters
    ----------
    sun_elevation : torch.Tensor
        Elevation angle of the sun position in degrees [°].
    heliostat_height : torch.Tensor
        Height of the heliostat mirror (sum over facets) in meters.
    helisotat_spacing : torch.Tensor
        Spacing along north-south direction to the front heliostat row in meters.
    translation_to_target : torch.Tensor
        Translation in ENU coordinates from heliostat center to target center.  
    """ 
    device = sun_elevation.device
    assert 90 >= sun_elevation.item() >= 0, f"sun_elevation should be [0°, 90°], not: {sun_elevation.item()}."
    assert heliostat_height.item() > 0, f"heliostat_height should be positive, not: {heliostat_height.item()}."
    assert heliostat_spacing.item() > 0, f"heliostat_spacing should be positive, not {heliostat_spacing.item()}."
    assert translation_to_target.shape[0] >= 3, f"Expected translation_to_target to be given in ENU-coordinates, but given dimensions: {translation_to_target.dim()}."
    
    # 1) Compute elevation angle for target reflection direction, i.e. heliostat -> target
    e_translation, n_translation, u_translation = translation_to_target[0], translation_to_target[1], translation_to_target[2]
    horizontal_norm = torch.sqrt(e_translation**2 + n_translation**2)
    local_target_elevation = torch.atan2(u_translation, horizontal_norm)
    
    # 2) Compute the elevation angle of the heliostat surface normal
    sun_elevation = torch.deg2rad(sun_elevation)
    heliostat_normal_elevation = (local_target_elevation + sun_elevation) / 2
    
    # 3) Compute the elevation angle of the heliostat surface
    heliostat_surface_elevation = torch.deg2rad(torch.tensor(90.0, device=device)) - heliostat_normal_elevation 
    
    # 4) Compute factor of shaded surface area relative to total surface area
    step_1 = torch.tan(sun_elevation)
    step_2 = heliostat_spacing / heliostat_height
    step_3 = step_1 / torch.cos(heliostat_surface_elevation)
    step_4 = torch.tan(heliostat_surface_elevation) + step_1
    f_shaded = 1.0 - step_2 * step_3 / step_4
    
    # 5) Compute geometric condition for shading
    step_5 = heliostat_height * torch.sin(heliostat_surface_elevation)
    step_6 = heliostat_spacing * step_1
    step_7 = step_1 / (torch.tan(heliostat_surface_elevation) + step_1)
    check_shade = step_6 / step_5 * (1 - step_7)
    if 0 <= check_shade.item() <= 1:
        return torch.clamp(f_shaded, min=0.0, max=1.0).item()
    
    return 0.0


def compute_sigma_pixels(effective_distance: float, target_height: float, num_vertical_px: int, 
                         sunshape_spread: float = 4.65, slope_error: float = 1.0):
    """
    Compute the diffusion sigma in pixel units for blocking attenuation.
    
    Parameters
    ----------
    effective_distance : flaot
        Effective distance from heliostat surface origin to target center in meters.
    target_height : float
        Height of the target plane (dimension in up-direction) in meters.
    num_vertical_pixels : int
        Number of pixels in vertical direction for the modeled flux image.
    sunshape_spread : float
        Sunshape spread / Solar cone half-width (~4.65 mrad).
    slope_error_mrad : float 
        RMS slope error of heliostat surface (1.0 - 4.0 mrad).
    
    Returns:
    --------
    sigma_px : float
        Diffusion standard deviation in pixels.
    """
    # Convert to radians
    theta_sun_rad = sunshape_spread / 1000
    theta_slope_rad = slope_error / 1000
    
    # Effective angular spread
    theta_eff = (theta_sun_rad**2 + theta_slope_rad**2)**0.5
    
    # Physical diffusion width in meters
    sigma_m = effective_distance * theta_eff
    
    # Convert to pixels
    meter_per_px = target_height / num_vertical_px
    sigma_px = sigma_m / meter_per_px

    return sigma_px / 2


def apply_artificial_blocking(image: torch.Tensor, block_ratio: float = 0.3, threshold: float = 0.1, sigma: float = 10.0, strength: float = 1.0):
    """
    Applies artificial blocking to the lower portion of a heliostat focal spot image
    using exponential growth of blocking strength.

    Parameters
    ----------
    image : torch.Tensor
        2D tensor (H, W), normalized pixel intensities.
    block_ratio : float
        Fraction (0–1) of the focal spot height to block from the bottom.
    threshold : float
        Intensity threshold to detect focal spot extent.
    sigma : float
        Controls the exponential growth rate of the blocking.
    strength : float
        Maximum blocking strength to reach toward the bottom (0 = none, 1 = full).
    background_threshold : float
        Pixel intensities below this value are considered background and will not be attenuated.

    Returns
    -------
    torch.Tensor
        Image with blocking applied.
    """
    assert image.dim() == 2, "Image must be a 2D tensor (H, W)"
    assert 0.0 <= strength <= 1.0, "Strength must be in [0, 1]"
    H, W = image.shape

    # Normalize image
    image_min, image_max = image.min(), image.max()
    if (image_max - image_min) > 1e-10:
        image = (image - image_min) / (image_max - image_min)
    else:
        return image

    # Identify focal spot bounds
    mask = image > threshold
    row_indices = torch.any(mask, dim=1).nonzero().squeeze()
    if row_indices.numel() == 0:
        return image

    top_idx = row_indices[0].item()
    bottom_idx = row_indices[-1].item()
    
    # Compute mean intensity per row and its gradient
    row_means = image.mean(dim=1)  # shape: (H,)
    row_gradients = torch.gradient(row_means)[0]

    # Identify top and bottom indices from extrema in the gradient
    # top_idx = torch.argmax(row_gradients).item()
    # bottom_idx = torch.argmin(row_gradients).item()
    if bottom_idx <= top_idx:
        return image  # skip blocking if region is invalid
    
    focal_height = bottom_idx - top_idx + 1
    block_start = int(bottom_idx - block_ratio * focal_height)

    # Construct vertical attenuation profile
    y = torch.arange(H, device=image.device).unsqueeze(1)  # shape (H, 1)
    rel_y = (y - block_start).clamp(min=0).float()
    decay = 1 - torch.exp(-rel_y / sigma)  # from 0 to ~1
    attenuation = 1.0 - strength * decay
    attenuation_mask = attenuation.expand(-1, W)
    
    # original = image.clone().detach()
    # image = image * attenuation_mask
    # return image
    
    # Apply attenuation, then enforce row-wise minimum value
    attenuated_image = image * attenuation_mask
    row_mins = image.min(dim=1, keepdim=True).values  # shape: (H, 1)
    attenuated_image = torch.maximum(attenuated_image, row_mins.expand(-1, W))

    return attenuated_image


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


def gaussian_filter_2d(image: torch.Tensor, kernel_size: int = 7, sigma: float = 1.5):
    """
    Apply a Gaussian filter to a 2D flux image using depthwise convolution.

    Parameters
    ----------
    image : torch.Tensor
        A 2D tensor of shape [H, W] representing the flux image.
    kernel_size : int
        Size of the Gaussian kernel (must be odd).
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    torch.Tensor
        Smoothed image of shape [H, W].
    """
    while image.dim() > 2 and image.shape[0] == 1:
        image = image.squeeze(0)
    assert image.dim() == 2, ValueError(f"Input image must be singe 2D image (HxW), but has shape {image.shape}")

    # Create 2D Gaussian kernel
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()

    # Expand to match convolution input requirements: [B=1, C=1, H, W]
    image = image.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
    kernel = kernel.to(image.device).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, kH, kW]

    # Apply depthwise convolution (padding to preserve size)
    padding = kernel_size // 2
    smoothed = F.conv2d(image, kernel, padding=padding)

    return smoothed.squeeze(0).squeeze(0)  # shape: [H, W]


if __name__ == '__main__':
    
    elevations = torch.tensor([1.61, 2.09, 5.84, 10, 15, 20, 30, 40, 50, 60, 65], dtype=torch.float32)
    h0 = torch.tensor(3.22, dtype=torch.float32)
    
    h_spacings = torch.tensor([4.6, 4.6, 6.54, 11.76], dtype=torch.float32)
    p_receiver = torch.tensor([0.0, 0.0, 55.0], dtype=torch.float32)
    h_coords = torch.tensor([      [0.0, -29.6, 2.5], 
                                   [0.0, -61.8, 2.5],
                                   [0.0, -125.44, 2.5],
                                   [0.0, -243.49, 2.5]
                            ], dtype=torch.float32)
    h_translations = p_receiver - h_coords
    
    shading = torch.empty((h_spacings.shape[0], elevations.shape[0]), dtype=torch.float32)
    blocking = torch.empty((h_spacings.shape[0], elevations.shape[0]), dtype=torch.float32)
    b_and_s = torch.empty((h_spacings.shape[0], elevations.shape[0]), dtype=torch.float32)
    
    for e in range(elevations.shape[0]):
        elevation = elevations[e]
        for h in range(h_spacings.shape[0]):
            h_spacing = h_spacings[h]
            h_translation = h_translations[h]
            
            blocking[h, e] = estimate_blocking_factor(elevation, h0, h_spacing, h_translation)
            shading[h, e] = estimate_shading_factor(elevation, h0, h_spacing, h_translation)
            b_and_s[h, e] = torch.max(blocking[h, e], shading[h, e])
    
    print(blocking[3].tolist())  
    print(shading[3].tolist())    
    print(b_and_s[3].tolist())
    