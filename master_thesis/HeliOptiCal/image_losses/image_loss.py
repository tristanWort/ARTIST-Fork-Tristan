import torch
import torch.nn.functional as funct
import torchvision.transforms as transforms
from torchmetrics.functional.segmentation import hausdorff_distance
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn as nn
from PIL import Image
from pathlib import Path
import time
import matplotlib.pyplot as plt

def contour_difference(predictions, targets, threshold=0.5, sharpness=20.0):
    """
    Compute the pixel-wise difference between contours of prediction and target image batches.
    
    Parameters:
    - predictions: Tensor with shape (B, H, 256, 256), values ∈ [0, 1]
    - targets: Tensor with shape (B, H, 256, 256), values ∈ [0, 1]
    - threshold: Threshold for contour extraction
    - sharpness: Steepness of the soft threshold
    
    Returns:
    - Difference bitmaps tensor with shape (B, H, 256, 256)
    """
    batch_size, height, width, width2 = predictions.shape
    assert width == width2 == 256, "Width dimensions must be 256x256"
    assert predictions.shape == targets.shape, "Prediction and target shapes must match"
    
    # Initialize the output tensor for difference bitmaps
    device = predictions.device
    difference_maps = torch.zeros_like(predictions, device=device)
    
    # Process each image in the batch
    for b in range(batch_size):
        for h in range(height):
            # Extract individual images
            pred_img = predictions[b, h]  # Shape: (256, 256)
            target_img = targets[b, h]    # Shape: (256, 256)
            
            # Find contours using the provided function
            pred_contour = find_soft_contour_pytorch_vertical(pred_img, threshold, sharpness)  # Shape: (1, 256, 256)
            target_contour = find_soft_contour_pytorch_vertical(target_img, threshold, sharpness)  # Shape: (1, 256, 256)
            
            # Calculate absolute difference between contours
            diff = torch.abs(pred_contour - target_contour)  # Shape: (1, 256, 256)
            
            # Store result in output tensor
            difference_maps[b, h] = diff.squeeze(0)  # Remove channel dimension
    
    return difference_maps

def find_soft_contour_pytorch_vertical(tensor_img, threshold=0.5, sharpness=20.0):
    """
    Differentiable contour extraction for vertical edges. Two convolutions are applied.
    First a Sobel kernel is used for edge detection. Then contours are extracted via soft erosion.

    Parameters:
   - tensor_img: Tensor with shape (1, H, W) or (H, W), values ∈ [0, 1]
    - threshold: Threshold for "soft" binarization
    - sharpness: Steepness of the soft threshold (higher ≈ sharper)

    Returns:
    - Soft contour mask (1, H, W), values ∈ [0, 1]
    """
    if tensor_img.dim() == 2:  # if batch size is missing
        tensor_img = tensor_img.unsqueeze(0)  # → (1, H, W)

    # First perform convolution for edge detection.
    # Define a Sobel kernel for vertical edge detection
    sobel_kernel = torch.tensor(
        [[-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]],
        device=tensor_img.device).float().view(1, 1, 3, 3)

    # Define a vertical edge detection kernel
    '''
    kernel = torch.tensor([[1, 0, -1], 
                        [2, 0, -2], 
                        [1, 0, -1]], device=tensor_img.device).float().view(1, 1, 3, 3)
    '''

    # Apply padding to maintain dimensions
    padded = funct.pad(((tensor_img - threshold) * sharpness).unsqueeze(0), (1, 1, 1, 1), mode='replicate')  # (1, 1, H+2, W+2)

    # Compute the vertical gradient
    grad_out = funct.conv2d(padded, sobel_kernel).squeeze(0)  # → (1, H, W)

    # Apply sigmoid function to get the mask for vertical edges
    mask = torch.sigmoid(grad_out)

    # Secondly, perform soft erosion for contour detection.
    # Soft Thresholding (Sigmoid instead of hard threshold)
    binary = torch.sigmoid((tensor_img - threshold) * sharpness)  # ∈ (0,1)

    # "Soft Erosion" via mean of the 3x3 neighborhood
    kernel = torch.ones((1, 1, 3, 3), device=tensor_img.device) / 9.0

    # Apply padding to maintain dimensions
    padded = funct.pad(binary.unsqueeze(0), (1, 1, 1, 1), mode='replicate')  # (1, 1, H+2, W+2)

    # Compute the mean of the 3x3 neighborhood
    neighborhood_mean = funct.conv2d(padded, kernel).squeeze(0)  # → (1, H, W)

    # Difference between center and neighborhood → contour measure
    contour_strength = binary - neighborhood_mean

    # Soft contour mask via Sigmoid (differentiable)
    soft_contour = (torch.sigmoid(contour_strength * sharpness) - 0.5) * 2

    # Clamp values to [0, 1]
    soft_contour = soft_contour.clamp(0, 1)
    return soft_contour * mask  # Values in [0,1], Shape: (1, H, W)


def mse_by_pixel(
        image1: torch.Tensor,
        image2: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the pixel-wise mean squared error between two images.

    Parameters
    ----------
    image1 : torch.Tensor
        The first image.
    image2 : torch.Tensor
        The second image.

    Returns
    -------
    torch.Tensor
        The pixel-wise loss.
    """
    return torch.mean((image1 - image2) ** 2)

def gaussian_kernel(
        size:int,
        sigma: float
) -> torch.Tensor:
    """
    Creates a 2D Gaussian kernel.

    Parameters
    ----------
    size: int
        The size of the kernel.
    sigma: float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    torch.Tensor
        The Gaussian kernel.
    """
    coords = torch.arange(size).float() - size // 2
    g = torch.exp(- coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()  # Normalize the kernel
    return g.outer(g)  # 1D outer product to create a 2D kernel

def ssim_map(
        image1: torch.Tensor,
        image2: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5,
        c1: float = 1e-4,
        c2: float = 9e-4
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between two grayscale images given as tensors.

    Parameters:
        image1 : torch.Tensor
            First image tensor of shape [H, W]
        image2 : torch.Tensor
            Second image tensor of shape [H, W]
        window_size: int
            Size of the Gaussian window (default: 11)
        sigma: float
            Standard deviation for Gaussian kernel (default: 1.5)
        c1: (float)
            Stability constant (default: 1e-4)
        c2: (float)
            Stability constant (default: 9e-4)

    Returns:
        torch.Tensor
            SSIM index (scalar)
    """
    assert image1.shape == image2.shape, "Input images must have the same shape."

    # Create Gaussian kernel
    kernel = gaussian_kernel(window_size, sigma).to(image1.device)
    kernel = kernel.expand(1, 1, -1, -1)  # [1, 1, window_size, window_size]

    # Add batch and channel dimensions
    image1 = image1.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    image2 = image2.unsqueeze(0).unsqueeze(0)

    # Compute means
    mu1 = funct.conv2d(image1, kernel, padding=window_size // 2)
    mu2 = funct.conv2d(image2, kernel, padding=window_size // 2)
    # Auxiliary mean products for SSIM calculation
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = funct.conv2d(image1 ** 2, kernel, padding=window_size // 2) - mu1_sq
    sigma2_sq = funct.conv2d(image2 ** 2, kernel, padding=window_size // 2) - mu2_sq
    sigma12 = funct.conv2d(image1 * image2, kernel, padding=window_size // 2) - mu1_mu2

    # Compute SSIM
    ssim_nominator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    ssim_denominator = ((mu1_sq + mu2_sq + c1) *
                        (sigma1_sq + sigma2_sq + c2))
    ssim_map = ssim_nominator / ssim_denominator
    return ssim_map

import torch
import torch.nn.functional as F

def chamfer_distance_batch_optimized(
        batch1: torch.Tensor,
        batch2: torch.Tensor,
        downsample_factor: int = 4,
        max_points: int = 1024,
        use_mixed_precision: bool = True,
        empty_image_value: float = 0.0,  # Value to assign for empty predictions
        empty_threshold: float = 1e-6    # Threshold to consider an image empty
) -> torch.Tensor:
    """
    Optimized Chamfer Distance calculation between two image batches, with handling for empty predictions.
    
    Parameters
    -------
    batch1 : torch.Tensor
        First batch of image tensors, i.e. predicted images
    batch2 : torch.Tensor
        Second batch of image tensors, i.e. target images
    downsample_factor : int
        Factor by which to downsample the images
    max_points : int
        Maximum number of points to use per image
    use_mixed_precision : bool
        Whether to use mixed precision computation
    empty_image_value : float
        Value to return for empty prediction images (skips computation)
    empty_threshold : float
        Threshold below which a prediction image is considered empty
        
    Returns
    -------
    torch.Tensor
        The Chamfer distance, with empty_image_value for empty predictions
    """
    assert batch1.shape == batch2.shape, "Input images must have the same shape."
    B, H, W = batch1.shape
    device = batch1.device
    result_dtype = batch1.dtype
    
    # Set computation dtype
    compute_dtype = torch.float16 if use_mixed_precision else result_dtype
    
    # Create a mask for non-empty predictions
    max_values = batch1.reshape(B, -1).max(dim=1)[0]  # Get max value per image
    non_empty_mask = max_values > empty_threshold     # Shape: (B,)
    
    # If all images are empty, return a batch of empty_image_value
    if not torch.any(non_empty_mask):
        return torch.full((B,), empty_image_value, device=device, dtype=result_dtype)
    
    # Prepare result tensor with default empty_image_value
    chamfer_distances = torch.full((B,), empty_image_value, device=device, dtype=compute_dtype)
    
    # If no valid images, return early
    if torch.sum(non_empty_mask) == 0:
        return chamfer_distances.to(result_dtype)
    
    # Extract non-empty batches to process
    batch1_valid = batch1[non_empty_mask]  # Shape: (valid_count, H, W)
    batch2_valid = batch2[non_empty_mask]  # Shape: (valid_count, H, W)
    valid_indices = torch.where(non_empty_mask)[0]
    valid_count = valid_indices.shape[0]
    
    # Downsample images to reduce computation
    if downsample_factor > 1:
        H_ds = H // downsample_factor
        W_ds = W // downsample_factor
        batch1_ds = funct.interpolate(batch1_valid.unsqueeze(1), size=(H_ds, W_ds), mode='area').squeeze(1)
        batch2_ds = funct.interpolate(batch2_valid.unsqueeze(1), size=(H_ds, W_ds), mode='area').squeeze(1)
    else:
        batch1_ds = batch1_valid
        batch2_ds = batch2_valid
        H_ds, W_ds = H, W
    
    # Convert to computation dtype
    batch1_ds = batch1_ds.to(compute_dtype)
    batch2_ds = batch2_ds.to(compute_dtype)
    
    # Use soft-thresholding
    threshold = 0.1
    image1_weights = torch.relu(batch1_ds - threshold)
    image2_weights = torch.relu(batch2_ds - threshold)
    
    # Generate coordinate grids (only once)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H_ds, device=device, dtype=compute_dtype),
        torch.arange(W_ds, device=device, dtype=compute_dtype),
        indexing='ij'
    )
    # Scale coordinates to match original image dimensions
    grid_y = grid_y.float() * downsample_factor
    grid_x = grid_x.float() * downsample_factor
    coords = torch.stack((grid_y, grid_x), dim=-1)  # Shape: (H_ds, W_ds, 2)
    
    # Process images in parallel where possible
    image1_weights_flat = image1_weights.reshape(valid_count, H_ds * W_ds)  # Shape: (valid_count, H_ds*W_ds)
    image2_weights_flat = image2_weights.reshape(valid_count, H_ds * W_ds)  # Shape: (valid_count, H_ds*W_ds)
    coords_flat = coords.reshape(-1, 2)  # Shape: (H_ds*W_ds, 2)
    
    # Calculate Chamfer distances for each valid batch element
    for i, b_idx in enumerate(valid_indices):
        # Get weights for current batch item
        img1_weights = image1_weights_flat[i]  # Shape: (H_ds*W_ds,)
        img2_weights = image2_weights_flat[i]  # Shape: (H_ds*W_ds,)
        
        # Check for sufficient active pixels
        active_pixels1 = (img1_weights > 0).sum().item()
        active_pixels2 = (img2_weights > 0).sum().item()
        
        if active_pixels1 < 2 or active_pixels2 < 2:
            # Skip this sample - leave default empty_image_value
            continue
        
        # Sample a limited number of points to reduce computation
        num_points1 = min(max_points, active_pixels1)
        num_points2 = min(max_points, active_pixels2)
        
        # Sample the most important points
        _, idx1 = torch.topk(img1_weights, num_points1)  # Shape: (num_points1,)
        _, idx2 = torch.topk(img2_weights, num_points2)  # Shape: (num_points2,)
        
        # Get coordinates and weights for selected points
        points1 = coords_flat[idx1]  # Shape: (num_points1, 2)
        points2 = coords_flat[idx2]  # Shape: (num_points2, 2)
        weights1 = img1_weights[idx1].unsqueeze(1)  # Shape: (num_points1, 1)
        weights2 = img2_weights[idx2].unsqueeze(1)  # Shape: (num_points2, 1)
        
        try:
            # Safely compute pairwise distances
            max_weight1 = weights1.max()
            max_weight2 = weights2.max()
            
            # Skip if the weights are too small
            if max_weight1 <= 1e-8 or max_weight2 <= 1e-8:
                continue
                
            # Normalize weights for numerical stability
            norm_weights1 = weights1 / max_weight1
            norm_weights2 = weights2 / max_weight2
            
            # Apply normalized weights
            weighted_points1 = points1 * norm_weights1
            weighted_points2 = points2 * norm_weights2
            
            # Compute pairwise distances
            dists = torch.cdist(weighted_points1, weighted_points2, p=2)
            
            # Compute minimum distances in both directions
            min_dists_1_to_2 = torch.min(dists, dim=1)[0]  # Shape: (num_points1,)
            min_dists_2_to_1 = torch.min(dists, dim=0)[0]  # Shape: (num_points2,)
            
            # Clip to avoid extreme values
            min_dists_1_to_2 = torch.clamp(min_dists_1_to_2, 0, 1e3)
            min_dists_2_to_1 = torch.clamp(min_dists_2_to_1, 0, 1e3)
            
            # Compute Chamfer distance
            chamfer_dist = (torch.mean(min_dists_1_to_2) + torch.mean(min_dists_2_to_1)) / 2
            
            # Ensure the distance is valid
            if torch.isfinite(chamfer_dist):
                chamfer_distances[b_idx] = chamfer_dist
        except Exception:
            # In case of any unexpected errors, skip this sample
            continue
    
    # Return the distances tensor in original dtype
    return chamfer_distances.to(result_dtype)

def chamfer_distance_batch(
        batch1: torch.Tensor,
        batch2: torch.Tensor
) -> torch.Tensor:
        """
        Calculate the Chamfer Distance between two image batches.

        Parameters
        -------
        batch1 : torch.Tensor
            First batch of image tensors, i.e. predicted images
        batch2 : torch.Tensor
            Second batch of image tensors, i.e. target images

        Returns
        -------
        torch.Tensor
            The Chamfer distance
        """

        assert batch1.shape == batch2.shape, "Input images must have the same shape."
        B, H, W = batch1.shape

        # Use soft-thresholding instead of hard binarization
        threshold = 0.1  # Define a small threshold to keep gradients
        image1_weights = torch.relu(batch1 - threshold)  # Soft selection
        image2_weights = torch.relu(batch2 - threshold)

        # Generate coordinate grids for all pixels
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=batch1.device),
            torch.arange(W, device=batch1.device),
            indexing='ij'
        )
        coords = torch.stack((grid_y, grid_x), dim=-1).float()  # Shape: (H, W, 2)
        coords = coords.view(1, H * W, 2).expand(B, -1, -1)  # Shape: (B, H*W, 2)

        # Flatten pixel weights
        image1_weights = image1_weights.view(B, H * W, 1)
        image2_weights = image2_weights.view(B, H * W, 1)

        # Softly select active pixel positions
        image1_coords = coords * image1_weights  # Shape: (B, H*W, 2)
        image2_coords = coords * image2_weights  # Shape: (B, H*W, 2)

        # Create masks for nonzero pixels
        image1_mask = image1_weights.squeeze(-1) > 0  # Shape: (B, H*W)
        image2_mask = image2_weights.squeeze(-1) > 0  # Shape: (B, H*W)

        # Select only nonzero coordinates for each batch element
        chamfer_distances = []
        for b in range(B):
            coords1 = image1_coords[b][image1_mask[b]]  # Shape: (N1, 2)
            coords2 = image2_coords[b][image2_mask[b]]  # Shape: (N2, 2)

            # If either image is empty, return a large finite value
            if coords1.numel() == 0 or coords2.numel() == 0:
                chamfer_distances.append(torch.tensor(1e6, device=batch1.device, dtype=batch1.dtype))
                continue

            # Compute pairwise Euclidean distances
            dists_1_to_2 = torch.cdist(coords1, coords2, p=2)  # (N1, N2)
            dists_2_to_1 = torch.cdist(coords2, coords1, p=2)  # (N2, N1)

            # Compute Chamfer distance: mean of nearest neighbor distances
            chamfer_dist = (torch.mean(torch.min(dists_1_to_2, dim=1)[0]) +
                            torch.mean(torch.min(dists_2_to_1, dim=1)[0]))

            chamfer_distances.append(chamfer_dist)

        return torch.stack(chamfer_distances)  # Shape: (B,)

def chamfer_distance_single(
        image1: torch.Tensor,
        image2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Chamfer distance between two grayscale images given as tensors.

    Parameters
    ----------
    image1: torch.Tensor
        The first image tensor (grayscale).
    image2: torch.Tensor
        The second image tensor (grayscale).

    Returns
    -------
    torch.Tensor
        The Chamfer distance.
    """
    assert image1.shape == image2.shape, "Input images must have the same shape."

    # Convert images to binary masks with rows containing indices for
    # non-zero values.
    image1_mask = torch.nonzero(torch.sigmoid(image1), as_tuple=False).float()
    image2_mask = torch.nonzero(torch.sigmoid(image2), as_tuple=False).float()

    # If either image has no significant pixels, return infinity
    if image1_mask.shape[0] == 0 or image2_mask.shape[0] == 0:
        return torch.tensor(1e6, device=image1.device, dtype=image1.dtype)

    # Compute pairwise Euclidian distances (L2 norm)
    dists_1_to_2 = torch.cdist(image1_mask, image2_mask, p=2)
    dists_2_to_1 = torch.cdist(image2_mask, image1_mask, p=2)

    # Compute the Chamfer distance: mean of nearest neighbor distances in both directions
    chamfer_dist = (torch.mean(torch.min(dists_1_to_2, dim=1)[0]) +
                     torch.mean(torch.min(dists_2_to_1, dim=1)[0]))

    return chamfer_dist

def hausdorff_loss(
        image1: torch.Tensor,
        image2: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Hausdorff distance between two grayscale images given as tensors.

    Parameters
    ----------
    image1 : torch.Tensor
        The first image tensor (grayscale).
    image2 : torch.Tensor
        The second image tensor (grayscale).

    Returns
    -------
    torch.Tensor
        The Hausdorff distance.
    """
    assert image1.shape == image2.shape, "Input images must have the same shape."

    B, H, W = image1.shape

    # Use soft-thresholding instead of hard binarization
    threshold = 0.1  # Define a small threshold to keep gradients
    image1_weights = torch.relu(image1 - threshold)  # Soft selection
    image2_weights = torch.relu(image2 - threshold)

    # Generate coordinate grids for all pixels
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=image1.device),
        torch.arange(W, device=image1.device),
        indexing='ij'
    )
    coords = torch.stack((grid_y, grid_x), dim=-1).float()  # Shape: (H, W, 2)
    coords = coords.view(1, H * W, 2).expand(B, -1, -1)  # Shape: (B, H*W, 2)

    # Flatten pixel weights
    image1_weights = image1_weights.view(B, H * W, 1)
    image2_weights = image2_weights.view(B, H * W, 1)

    # Softly select active pixel positions
    image1_coords = coords * image1_weights  # Shape: (B, H*W, 2)
    image2_coords = coords * image2_weights  # Shape: (B, H*W, 2)

    # Create masks for nonzero pixels
    image1_mask = image1_weights.squeeze(-1) > 0  # Shape: (B, H*W)
    image2_mask = image2_weights.squeeze(-1) > 0  # Shape: (B, H*W)

    # Select only nonzero coordinates for each batch element
    hausdorff_distances = []
    for b in range(B):

        coords1 = image1_coords[b][image1_mask[b]]  # Shape: (N1, 2)
        coords2 = image2_coords[b][image2_mask[b]]  # Shape: (N2, 2)

        # If either image is empty, return a large finite value
        if coords1.numel() == 0 or coords2.numel() == 0:
            hausdorff_distances.append(torch.tensor(1e6, device=image1.device, dtype=image1.dtype))
            continue

        # Compute pairwise Euclidean distances
        dists_1_to_2 = torch.cdist(coords1, coords2, p=2)  # (N1, N2)
        dists_2_to_1 = torch.cdist(coords2, coords1, p=2)  # (N2, N1)

        # Compute the Hausdorff distance: maximum of nearest neighbor distances in both directions
        hausdorff_dist = torch.max(torch.max(torch.min(dists_1_to_2, dim=1)[0]),
                                   torch.max(torch.min(dists_2_to_1, dim=1)[0]))

        hausdorff_distances.append(hausdorff_dist)

    return torch.stack(hausdorff_distances).mean()

""" !Computationally expensive! Use sliced version. """
class EarthMoversDistance(nn.Module):
    """
    A differentiable implementation of the Earth Mover's Distance (Wasserstein-1 distance)
    between two 2D image tensors using Sinkhorn iteration.

    This implementation is based on the Sinkhorn algorithm which provides a differentiable
    approximation to the optimal transport problem.
    """

    def __init__(self, eps=0.1, max_iter=100, reduction='mean'):
        """
        Initialize the EMD module.

        Parameters
        ----------
        eps : float
            Regularization parameter for Sinkhorn algorithm. Lower values
            give more accurate EMD but may be numerically unstable.
        max_iter : int
            Maximum number of Sinkhorn iterations.
        reduction : str
            'mean', 'sum' or 'none'. Determines the reduction to apply to the output.
        """
        super(EarthMoversDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Calculate the differentiable EMD between predicted and target images.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted image tensor of shape (batch_size, channels, height, width)
        target: torch.Tensor
            Target image tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        torch.Tensor
            The EMD loss
        """
        # Ensure inputs are at least 4D (batch_size, channels, height, width)
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif pred.dim() == 3:
            pred = pred.unsqueeze(1)  # Add channel dimension

        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.dim() == 3:
            target = target.unsqueeze(1)

        batch_size, channels, height, width = pred.shape

        # Normalize the distributions so they sum to 1
        # Adding a small epsilon to avoid division by zero
        pred_flat = pred.view(batch_size, channels, -1)
        target_flat = target.view(batch_size, channels, -1)

        pred_norm = pred_flat / (torch.sum(pred_flat, dim=2, keepdim=True) + 1e-10)
        target_norm = target_flat / (torch.sum(target_flat, dim=2, keepdim=True) + 1e-10)

        # Compute cost matrix based on pixel locations
        # Creating coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=pred.device, dtype=torch.float32),
            torch.arange(width, device=pred.device, dtype=torch.float32),
            indexing='ij'
        )

        # Flatten coordinates
        y_flat = y_grid.reshape(-1)
        x_flat = x_grid.reshape(-1)

        # Compute pairwise squared distances between all pixel locations
        # This creates a cost matrix of shape (height*width, height*width)
        i_coords = torch.stack([y_flat, x_flat], dim=1).unsqueeze(1)  # (h*w, 1, 2)
        j_coords = torch.stack([y_flat, x_flat], dim=1).unsqueeze(0)  # (1, h*w, 2)
        cost_matrix = torch.sum((i_coords - j_coords) ** 2, dim=2)  # (h*w, h*w)

        # Initialize results tensor
        emd_loss = torch.zeros(batch_size, channels, device=pred.device)

        # Iterate over each image in the batch and each channel
        for b in range(batch_size):
            for c in range(channels):
                a = pred_norm[b, c]  # Source distribution
                b_dist = target_norm[b, c]  # Target distribution

                # Sinkhorn algorithm to compute optimal transport
                u = torch.zeros_like(a)
                v = torch.zeros_like(b_dist)

                # Pre-compute K
                K = torch.exp(-cost_matrix / self.eps)

                # Sinkhorn iterations
                for _ in range(self.max_iter):
                    u = torch.log(a + 1e-10) - torch.log(torch.matmul(K, torch.exp(v)) + 1e-10)
                    v = torch.log(b_dist + 1e-10) - torch.log(torch.matmul(K.t(), torch.exp(u)) + 1e-10)

                # Compute the transport plan
                P = torch.exp(u.unsqueeze(1) + v.unsqueeze(0)) * K

                # Compute the EMD (cost times the transport plan)
                emd_value = torch.sum(P * cost_matrix)
                emd_loss[b, c] = emd_value

        # Apply reduction
        if self.reduction == 'mean':
            return emd_loss.mean()
        elif self.reduction == 'sum':
            return emd_loss.sum()
        else:  # 'none'
            return emd_loss

""" Non-deterministic. """
class SlicedWassersteinDistance(nn.Module):
    """
    An alternative implementation using the Sliced Wasserstein Distance,
    which is more efficient for 2D images and is naturally differentiable.

    This approximates the Wasserstein distance by computing 1D Wasserstein
    distances along random projections.
    """

    def __init__(self, num_projections=50, reduction='mean'):
        """
        Initialize the SWD module.

        Parameters
        -------
            num_projections: int
                Number of random projections to use
            reduction: str
                'mean', 'sum' or 'none'
        """
        super(SlicedWassersteinDistance, self).__init__()
        self.num_projections = num_projections
        self.reduction = reduction

    def forward(self, batch1, batch2):
        """
        Calculate the Sliced Wasserstein Distance between predicted and target images.

        Parameters
        -------
        batch1 : torch.Tensor
            First batch of image tensors, i.e. predicted images
        batch2 : torch.Tensor
            Second batch of image tensors, i.e. target images

        Returns
        -------
        torch.Tensor
            The SWD loss
        """

        # Ensure inputs are at least 4D (batch_size, channels, height, width)
        if batch1.dim() == 2:
            batch1 = batch1.unsqueeze(0).unsqueeze(0)
        elif batch1.dim() == 3:
            batch1 = batch1.unsqueeze(1)

        if batch2.dim() == 2:
            batch2 = batch2.unsqueeze(0).unsqueeze(0)
        elif batch2.dim() == 3:
            batch2 = batch2.unsqueeze(1)

        assert batch1.shape == batch2.shape, "Given batches must have the same size and shape."
        batch_size, channels, height, width = batch1.shape

        # Flatten spatial dimensions
        batch1_flat = batch1.view(batch_size, channels, -1)
        batch2_flat = batch2.view(batch_size, channels, -1)

        # Initialize result tensor
        swd = torch.zeros(batch_size, channels, device=batch1.device)

        for b in range(batch_size):
            for c in range(channels):
                # Generate random unit vectors for projection
                theta = torch.randn(self.num_projections, height * width, device=batch1.device)
                theta = theta / torch.norm(theta, dim=1, keepdim=True)

                # Project the distributions onto the random directions
                proj_batch1 = torch.matmul(batch1_flat[b, c], theta.t())  # (num_projections,)
                proj_batch2 = torch.matmul(batch2_flat[b, c], theta.t())  # (num_projections,)

                # Sort the projections
                proj_batch1, _ = torch.sort(proj_batch1, dim=0)
                proj_batch2, _ = torch.sort(proj_batch2, dim=0)

                # Compute the L2 Wasserstein distance along each projection
                wasserstein_dist = torch.sum((proj_batch1 - proj_batch2) ** 2, dim=0)

                # Average over all projections
                swd[b, c] = torch.mean(wasserstein_dist)

        # Apply reduction
        if self.reduction == 'mean':
            return torch.sqrt(swd).mean()
        elif self.reduction == 'sum':
            return torch.sqrt(swd).sum()
        else:  # 'none'
            return torch.sqrt(swd)

""" Not working properly yet """
def sinkhorn_distance(
        image1: torch.Tensor,
        image2: torch.Tensor,
        epsilon: float = 1e-3,
        n_iters: int = 100,
) -> torch.Tensor:
    """
    Compute the Sinkhorn approximation of the Wasserstein distance between two grayscale images.
    # Todo: Compare function to class output.

    Parameters
    ----------
    image1
    image2
    epsilon
    n_iters

    Returns
    -------

    """
    # Compute the pairwise cost (L2 distance)
    cost_matrix = torch.cdist(image1.unsqueeze(0), image2.unsqueeze(0), p=2).squeeze(0)  # Shape [N, M]

    # Initialize dual potentials
    u = torch.zeros_like(image1)  # Shape[N]
    v = torch.zeros_like(image2)  # Shape[M]

    for _ in range(n_iters):
        u = -epsilon * (torch.logsumexp((-cost_matrix + v.unsqueeze(0)) / epsilon, dim=1) - v) + u.mean()
        v = -epsilon * (torch.logsumexp((-cost_matrix + u.unsqueeze(1)) / epsilon, dim=0) - u) + v.mean()

    # Compute the final transport cost
    transport_cost = torch.sum(u + image1) + torch.sum(v + image2)
    return transport_cost


if __name__ == '__main__':

    load_image1 = Path(r'C:\Users\wort_tr\Documents\00_Masterarbeit\06_PAINT\AM35\Calibration\131204-flux.png')
    load_image2 = Path(r'C:\Users\wort_tr\Documents\00_Masterarbeit\06_PAINT\AM35\Calibration\134381-flux.png')
    load_image3 = Path(r'C:\Users\wort_tr\Documents\00_Masterarbeit\06_PAINT\AM35\Calibration\102880-flux.png')
    original_img1 = Image.open(load_image1)
    original_img1 = (transforms.ToTensor()(original_img1)).squeeze(0)
    original_img2 = Image.open(load_image2)
    original_img2 = (transforms.ToTensor()(original_img2)).squeeze(0)
    original_img3 = Image.open(load_image3)
    original_img3 = (transforms.ToTensor()(original_img3)).squeeze(0)

    print('images loaded')
    hl = hausdorff_loss(original_img1.unsqueeze(0), original_img2.unsqueeze(0))
    print(hl)
    input()
    # batch1 = torch.stack([original_img1, original_img2, original_img3,original_img1, original_img2, original_img3])
    # batch2 = torch.stack([original_img2, original_img1, original_img3,original_img1, original_img2, original_img3])
    # start_time = time.time()
    # chamfer_dist = chamfer_distance(original_img1, original_img2)
    # end_time = time.time()
    # chamfer_dist2 = chamfer_distance(original_img1, original_img2)
    #
    #
    # print(f'Chamfer Distance: {chamfer_dist.tolist()}')
    # print(f'Time taken: {end_time - start_time:.4f} seconds')
    #
    # print(f'Chamfer Distance 2: {chamfer_dist2.item():.4f}')
    # input()
    # sliced_emd_los_fnc = SlicedWassersteinDistance(num_projections=100)
    # print('Calculate sliced EMD...')
    # sliced_emd_loss12 = sliced_emd_los_fnc(original_img1, original_img2)
    # sliced_emd_loss13 = sliced_emd_los_fnc(original_img1, original_img3)
    # sliced_emd_loss21 = sliced_emd_los_fnc(original_img2, original_img1)
    # sliced_emd_loss31 = sliced_emd_los_fnc(original_img3, original_img1)
    # sliced_emd_loss23 = sliced_emd_los_fnc(original_img2, original_img3)
    # sliced_emd_loss32 = sliced_emd_los_fnc(original_img3, original_img2)
    # sliced_emd_loss11 = sliced_emd_los_fnc(original_img1, original_img1)
    #
    # print(f'Sliced EMD Loss 12: {sliced_emd_loss12.item():.4f}')
    # print(f'Sliced EMD Loss 13: {sliced_emd_loss13.item():.4f}')
    # print(f'Sliced EMD Loss 21: {sliced_emd_loss21.item():.4f}')
    # print(f'Sliced EMD Loss 31: {sliced_emd_loss31.item():.4f}')
    # print(f'Sliced EMD Loss 23: {sliced_emd_loss23.item():.4f}')
    # print(f'Sliced EMD Loss 32: {sliced_emd_loss32.item():.4f}')
    # print(f'Sliced EMD Loss 11: {sliced_emd_loss11.item():.4f}')
    #
    #
    # input()
    #
    # chamfer_distance = chamfer_distance(original_img1, original_img2)
    ssim = ssim_map(original_img1, original_img2)

    ssim = ssim.squeeze(0).squeeze(0)

    ssim_torch = StructuralSimilarityIndexMeasure()
    original_img1 = original_img1.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    original_img2 = original_img2.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    ssim_torch_value = ssim_torch(original_img1, original_img2)

    print("SSIM from custom function: ", ssim.mean().item())
    print("SSIM from torchmetrics: ", ssim_torch_value)

    # original_img1 = (transforms.ToTensor()(original_img1)).squeeze(0)


    window_size = 11
    sigma = 1.5
    gauss_kernel = gaussian_kernel(window_size, sigma)
    gauss_kernel = gauss_kernel.expand(1, 1, -1, -1)
    mu1 = funct.conv2d(original_img1, gauss_kernel, padding=window_size // 2)
    mu1_sq = mu1 ** 2

    sigma1_sq = funct.conv2d(original_img1 ** 2, gauss_kernel, padding=window_size // 2) - mu1_sq

    # Squeeze
    original_img1 = original_img1.squeeze(0).squeeze(0)
    mu1 = mu1.squeeze(0).squeeze(0)
    sigma1_sq = sigma1_sq.squeeze(0).squeeze(0)

    # Plot the original image and the mu1 tensor as heat-maps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_img1.detach().numpy(), cmap='inferno')
    axes[0].set_title('Original Image 131204')
    axes[0].axis('off')
    axes[1].imshow(original_img2.cpu().detach().numpy(), cmap='inferno')
    axes[1].set_title('Original Image 134381')
    axes[1].axis('off')


    axes[2].imshow(ssim.detach().numpy(), cmap='inferno')
    axes[2].set_title(f'SSIM Map with Mean: {ssim.mean().item():.4f}')
    axes[2].axis('off')

    plt.show()

