import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def generate_random_point_cloud(num_points, width, height):
    """
    Generate a random point cloud with points inside a specified bitmap size.
    
    Args:
        num_points: Number of points in the point cloud
        width: Width of the bitmap
        height: Height of the bitmap
        
    Returns:
        Tensor of shape (num_points, 2) representing the 2D point cloud
    """
    points = torch.zeros((num_points, 2))
    
    # Generate random coordinates within the bitmap dimensions
    points[:, 0] = torch.rand(num_points) * width  # x-coordinates
    points[:, 1] = torch.rand(num_points) * height  # y-coordinates
    
    return points

def transform_point_cloud(points, translation, rotation, rotation_center=None):
    """
    Apply translation and rotation to a point cloud.
    
    Args:
        points: Tensor of shape (N, 2) representing the 2D point cloud
        translation: Tensor of shape (2,) representing the translation vector
        rotation: Tensor representing the rotation angle in radians
        rotation_center: Optional center of rotation. If None, rotate around origin
        
    Returns:
        Transformed point cloud of shape (N, 2)
    """
    # Make a copy of the points to avoid modifying the original
    transformed_points = points.clone()
    
    # If rotation center is provided, shift points to make that the origin
    if rotation_center is not None:
        transformed_points = transformed_points - rotation_center
    
    # Create rotation matrix
    cos_theta, sin_theta = torch.cos(rotation), torch.sin(rotation)
    rot_matrix = torch.stack([
        torch.stack([cos_theta, -sin_theta]),
        torch.stack([sin_theta, cos_theta])
    ])
    
    # Apply rotation
    transformed_points = transformed_points @ rot_matrix.T
    
    # If rotation center is provided, shift points back
    if rotation_center is not None:
        transformed_points = transformed_points + rotation_center
    
    # Apply translation
    transformed_points = transformed_points + translation.unsqueeze(0)
    
    return transformed_points

"""My Chamfer loss function"""
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
        batch1_ds = F.interpolate(batch1_valid.unsqueeze(1), size=(H_ds, W_ds), mode='area').squeeze(1)
        batch2_ds = F.interpolate(batch2_valid.unsqueeze(1), size=(H_ds, W_ds), mode='area').squeeze(1)
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


def plot_point_clouds(ax, target_pts, source_pts, title):
    """Helper function to plot point clouds"""
    ax.scatter(target_pts[:, 0].detach().cpu().numpy(), 
               target_pts[:, 1].detach().cpu().numpy(),
               c='blue', marker='o', s=100, label='Target')
    
    ax.scatter(source_pts[:, 0].detach().cpu().numpy(), 
               source_pts[:, 1].detach().cpu().numpy(),
               c='red', marker='x', s=100, label='Source')
    
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(title)
    ax.legend()


def main():
    # Parameters for the bitmap and point clouds
    bitmap_width, bitmap_height = 256, 256
    num_points = 5  # 5 points in each point cloud
    
    # Generate random target point cloud (static)
    target_points = generate_random_point_cloud(num_points, bitmap_width, bitmap_height).to(device)
    
    # Generate a separate random source point cloud (to be aligned)
    source_points = generate_random_point_cloud(num_points, bitmap_width, bitmap_height).to(device)
    
    # Calculate center of source point cloud (for rotation reference)
    source_center = torch.mean(source_points, dim=0)
    
    # Parameters to optimize: translation and rotation
    # Initialize with some offset to make the problem interesting
    init_translation = torch.tensor([20.0, -15.0], device=device, requires_grad=True)
    init_rotation = torch.tensor(0.3, device=device, requires_grad=True)  # in radians
    
    # Optimizer
    optimizer = optim.Adam([init_translation, init_rotation], lr=0.01)
    
    # Number of optimization steps
    num_steps = 200
    
    # Lists to store losses and parameters for visualization
    losses = []
    translations_x = []
    translations_y = []
    rotations = []
    
    # Store the initial transformed source for visualization
    initial_transformed = transform_point_cloud(
        source_points, init_translation, init_rotation, source_center
    )
    
    # Optimization loop
    print("Starting optimization...")
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Transform the source point cloud with current parameters
        transformed_source = transform_point_cloud(
            source_points, init_translation, init_rotation, source_center
        )
        
        # Calculate Chamfer distance using the user's function
        loss = chamfer_distance_batch(transformed_source.unsqueeze(0), target_points.unsqueeze(0))
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        
        # Store parameters and loss for visualization
        losses.append(loss.item())
        translations_x.append(init_translation[0].item())
        translations_y.append(init_translation[1].item())
        rotations.append(init_rotation.item())
        
        if step % 20 == 0 or step == num_steps - 1:
            print(f"Step {step}, Loss: {loss.item():.6f}, "
                  f"Translation: ({init_translation[0].item():.2f}, {init_translation[1].item():.2f}), "
                  f"Rotation: {init_rotation.item():.4f} rad")
    
    # Get final transformed point cloud
    final_transformed_source = transform_point_cloud(
        source_points, init_translation, init_rotation, source_center
    )
    
    # Visualize results
    fig = plt.figure(figsize=(15, 10))
    
    # Plot the original point clouds
    ax1 = fig.add_subplot(2, 2, 1)
    plot_point_clouds(ax1, target_points, source_points, 'Original Point Clouds')
    
    # Plot the initial transformation
    ax2 = fig.add_subplot(2, 2, 2)
    plot_point_clouds(ax2, target_points, initial_transformed, 'Initial Transformation')
    
    # Plot the final aligned point clouds
    ax3 = fig.add_subplot(2, 2, 3)
    plot_point_clouds(ax3, target_points, final_transformed_source, 'Aligned Point Clouds')
    
    # Plot optimization progress
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(losses, label='Chamfer Distance')
    ax4.set_xlabel('Optimization Step')
    ax4.set_ylabel('Loss')
    ax4.set_title('Optimization Progress')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot parameter changes
    plt.figure(figsize=(10, 6))
    plt.plot(translations_x, label='Translation X')
    plt.plot(translations_y, label='Translation Y')
    plt.plot(rotations, label='Rotation (rad)')
    plt.xlabel('Optimization Step')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Changes During Optimization')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nFinal parameters:")
    print(f"Translation: ({init_translation[0].item():.2f}, {init_translation[1].item():.2f})")
    print(f"Rotation: {init_rotation.item():.4f} rad ({init_rotation.item() * 180 / np.pi:.2f} degrees)")
    
    return target_points, source_points, init_translation, init_rotation

if __name__ == "__main__":
    target_points, source_points, translation, rotation = main()
    