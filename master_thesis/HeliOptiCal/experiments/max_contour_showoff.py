import pathlib
import torch
import cv2
import numpy as np
import requests
import tempfile
import os
import sys
import h5py
from matplotlib import pyplot as plt, animation
from scipy.ndimage import zoom
from PIL import Image
import torchvision as thv
from torchmetrics.functional import structural_similarity_index_measure as ssim_loss

import paint.util.paint_mappings as mappings
from paint import PAINT_ROOT
from paint.data.stac_client import StacClient
from paint.preprocessing.focal_spot_extractor import detect_focal_spot
from paint.preprocessing.target_cropper import crop_image_with_template_matching

# Add local artist path for raytracing with multiple parallel heliostats.
artist_repo = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, artist_repo)  
from artist.util.scenario import Scenario
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.utils import get_center_of_mass
from artist.raytracing import raytracing_utils

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, repo_path)
from HeliOptiCal.utils.util import (calculate_intersection,
                                    normalize_images, 
                                    normalize_and_interpolate,
                                    get_bitmap_indices_from_center_coordinates, 
                                    get_rigid_body_kinematic_parameters_from_scenario)
from HeliOptiCal.utils.util_simulate import gaussian_filter_2d
from HeliOptiCal.image_losses.image_loss import find_soft_contour_pytorch_vertical, sdf_loss, dice_loss
 
 
def load_model_from_url(url: str):
    """
    Download the model checkpoint from the given URL and load it directly without storing it permanently.

    Parameters
    ----------
    url : str
        URL of the model checkpoint.

    Returns
    -------
    torch.jit.ScriptModule
        Loaded PyTorch model.

    Raises
    ------
    Exception
        If the model cannot be loaded or the download fails.
    """
    print(f"Downloading checkpoint from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        print("Checkpoint downloaded successfully. Loading the model...")
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.flush()
            try:
                model = torch.jit.load(temp_file.name, map_location="cpu")
                print("Model loaded successfully.")
                return model
            except Exception as e:
                print(f"Failed to load the model: {e}")
                raise
    else:
        print(f"Failed to download the checkpoint. Status code: {response.status_code}")
        response.raise_for_status()
        
        
def crop_image(img_path: pathlib.Path, corners_file: pathlib.Path, output_size=(256, 256)):
    """
    Crop image along corners provided in a text-file.
    """
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # shape: [H, W]
    points = np.loadtxt(corners_file, delimiter=",", skiprows=1, dtype=np.float32)
    
    assert (points.shape[1] == 2 and points.ndim == 2), f"Wrong format of points: {points.shape}"
    
    # Create mapper for target size 
    width, height = output_size
    pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    # Compute and apply perspective transform
    M = cv2.getPerspectiveTransform(points, pts_dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped


def plot_center_cross(center_indices: (int, int), cross_size: int = 5, cross_color: str = "red", cross_linewidth: float = 0.5):
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
    

def save_bitmap(bitmap:torch.Tensor, save_under: pathlib.Path, target_size=(256, 256), center_indices=None):
    
    if bitmap.dim() == 4:
        bitmap = bitmap.squeeze(0).squeeze(0)
    elif bitmap.dim() == 3:
        bitmap = bitmap.squeeze(0)
        
    width_in_pixels, height_in_pixels = target_size
    dpi = 100  # Can be any value
    width_in_inches = width_in_pixels / dpi
    height_in_inches = height_in_pixels / dpi

    plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
    plt.imshow(bitmap.cpu().detach(), cmap="gray")
    
    if center_indices is not None:
        plot_center_cross(center_indices)    
        
    plt.axis("off")  # Hides both x and y axes
    plt.savefig(save_under, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()


def save_bitmaps_as_gif(bitmaps: [torch.Tensor], save_under: pathlib.Path):
    fig, ax = plt.subplots()
    im = ax.imshow(bitmaps[0], cmap="gray")
    ax.axis("off")
    
    def update(frame):
        im.set_array(bitmaps[frame])
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=len(bitmaps), blit=True)
    plt.tight_layout()
    plt.axis("off")  # Hides both x and y axes
    ani.save(save_under, writer='pillow', fps=10, dpi=100)
    plt.close(fig)
    
    
def undo_flux_geometric_scaling(composite_image, flux_scale, downsample_shape, orig_shape):
    """
    Reverse the geometric zoom and downsampling of a flux image inside a superposed background image.
    
    Parameters:
        composite_image (np.ndarray): The image containing the placed flux (e.g., composite_flux or result)
        flux_scale (float): Scaling factor used for zoom (e.g., 3.0)
        downsample_shape (tuple): (H, W) used for downsampling before zoom
        orig_shape (tuple): Original flux image shape (H, W)
    
    Returns:
        np.ndarray: The recovered flux image with focal spot at original geometric size
    """
    # 1. Compute size of zoomed flux image
    zoomed_h = int(downsample_shape[0] * flux_scale)
    zoomed_w = int(downsample_shape[1] * flux_scale)
    
    # 2. Estimate placement region in composite image
    # Compute center of composite canvas
    canvas_h, canvas_w = composite_image.shape
    center_y = canvas_h // 2
    center_x = canvas_w // 2

    # Get top-left corner for extracting zoomed flux
    y1 = center_y - zoomed_h // 2
    x1 = center_x - zoomed_w // 2
    y2 = y1 + zoomed_h
    x2 = x1 + zoomed_w

    # Crop out the zoomed flux from composite
    cropped_zoom = composite_image[y1:y2, x1:x2]

    # 3. Undo zoom
    flux_downsampled_est = zoom(cropped_zoom, zoom=1 / flux_scale)

    # 4. Resize back to original resolution
    flux_restored = np.array(Image.fromarray(flux_downsampled_est).resize(
        (orig_shape[1], orig_shape[0]), Image.BILINEAR
    ))

    return flux_restored
  

def normalize_on_mean(bitmap: torch.Tensor):
    """
    Normalize bitmap pixels on their mean.
    """
    assert bitmap.dim() == 2, "Bitmap dimensions should be 2, not {bitmap.dim()}"
    min_value = bitmap.min()
    bitmap = (bitmap - min_value)
    mean_value = bitmap.mean()    
    return bitmap / mean_value
    

def processs_image(directory: pathlib.Path, img_file_name: str, corners_file_name: str, tower="stj"): 

    if tower.upper() == "STJ":
        downscale = (64, 64)
    else:
        downscale = (128, 128)
        
    image = cv2.imread(directory/img_file_name, cv2.IMREAD_GRAYSCALE)  # shape: [H, W]
    restored_flux = undo_flux_geometric_scaling(image, 3.0, downscale, (960, 960))
    restored_tensor = (torch.tensor(restored_flux, dtype=torch.float32))
    
    save_bitmap(restored_tensor, directory/f'gt_restored.png')
    # norm_restored = normalize_images(restored_tensor).unsqueeze(0)
    # print(norm_restored.shape)
    
    import torch.nn.functional as F
    restored_tensor = gaussian_filter_2d(restored_tensor, sigma=5.0)
    img_transform = thv.transforms.Compose([
        thv.transforms.Resize((256, 256)),
        # Remove single channel.
        thv.transforms.Lambda(lambda image: image.squeeze(0)),
        # Try to remove background (i.e. make background dark).
        thv.transforms.Lambda(lambda image: torch.clip(
            image - image.median(),
            min=0,
        )),
    ]) 
    clean_flux = img_transform(restored_tensor.unsqueeze(0))
    # norm_clean = normalize_on_mean(clean_flux.squeeze(0))
    norm_clean = normalize_images(clean_flux).squeeze(0)
    save_bitmap(norm_clean, directory/f'gt_restored_background_removed.png')
    return norm_clean

    # contour = find_soft_contour_pytorch_vertical(resized.squeeze(0).squeeze(0), threshold=0.6, sharpness=70)
    # save_bitmap(contour, directory/f'contour_pred.png')
    # return contour


def map_restored_flux_pixel_to_composite(pixel_coords, flux_scale, tower):
    """
    Map pixel coordinates from the restored original flux image (e.g., 960x960)
    back to the composite image (e.g., 256x256) that contained the zoomed flux image.

    Parameters:
        pixel_coords (tuple): (y, x) pixel coordinates in the restored flux image (orig_shape)
        flux_scale (float): The geometric zoom factor used during augmentation
        downsample_shape (tuple): Shape (H, W) used in downsampling before zoom
        composite_shape (tuple): Shape (H, W) of the composite image (e.g., 256x256)

    Returns:
        tuple: (y, x) coordinates in the composite image space
    """
    orig_y, orig_x = pixel_coords
    
    if tower.upper() == "STJ":
        downsample_shape = (64, 64)
        composite_shape = (256, 256)
    else:
        downsample_shape = (128, 128)
        composite_shape = (1024, 1376)
        
    downsample_h, downsample_w = downsample_shape
    comp_h, comp_w = composite_shape

    # Step 1: Map from original (960x960) to downsampled (e.g., 64x64)
    scale_y_down = downsample_h / 256
    scale_x_down = downsample_w / 256
    down_y = orig_y * scale_y_down
    down_x = orig_x * scale_x_down

    # Step 2: Apply zoom (e.g., 3x)
    zoom_y = down_y * flux_scale
    zoom_x = down_x * flux_scale

    # Step 3: Center placement in composite image
    center_y = comp_h // 2
    center_x = comp_w // 2
    top_left_y = center_y - int((downsample_h * flux_scale) // 2)
    top_left_x = center_x - int((downsample_w * flux_scale) // 2)

    # Step 4: Compute composite pixel coordinates
    composite_y = top_left_y + zoom_y
    composite_x = top_left_x + zoom_x

    return composite_y, composite_x


def apply_gaussian_find_contours(input_bitmaps, sharpness=20, threshold=0.1, sigma_in=2.5, sigma_out=2.5, num_interpolations=0):
    """
    Function for applying gaussian blurring to input bitmap and returning the upper contour.
    """
    # First normalize inputs
    input_bitmaps = normalize_and_interpolate(input_bitmaps, num_interpolations).squeeze(0)
    gauss_bitmaps = torch.empty_like(input_bitmaps)
    gauss_contours = torch.empty_like(input_bitmaps)
    for i in range(input_bitmaps.shape[0]):
        # Apply Gaussian filter on inputs, save and return
        gauss_img = gaussian_filter_2d(input_bitmaps[i], sigma=sigma_in)
        gauss_bitmaps[i] = normalize_images(gauss_img.unsqueeze(0))
        # Find contours, apply Gaussian filter on output
        contour = find_soft_contour_pytorch_vertical(gauss_bitmaps[i], threshold=threshold, sharpness=sharpness)
        gauss_contours[i] = gaussian_filter_2d(contour.squeeze(0), sigma=sigma_out)
    # Normalize output
    contours = normalize_and_interpolate(gauss_contours, num_interpolations).squeeze(0)
    return contours


def weight_loss(images1: torch.Tensor, images2: torch.Tensor, target_areas: list(), threshold=0.1):
    
    assert images1.shape == images2.shape, f"shapes must be equal: {images1.shape, images2.shape}"
    if images1.dim() == 2:
        images1 = images1.unsqueeze(0)
        images2 = images2.unsqueeze(0)
    elif images1.dim() == 4:    
        images1 = images1.squeeze(0)
        images2 = images2.squeeze(0)
        
    device = images1.device
    center_of_mass1 = []
    center_of_mass2 = []
    
    for i in range(images1.shape[0]):
        target_area =  target_areas[i]
        center_of_mass1.append(get_center_of_mass(bitmap=images1[i], 
                                                  target_center=target_area.center,
                                                  plane_e=target_area.plane_e,
                                                  plane_u=target_area.plane_u,
                                                  threshold=threshold,
                                                  device=device))
        center_of_mass2.append(get_center_of_mass(bitmap=images2[i], 
                                                  target_center=target_area.center,
                                                  plane_e=target_area.plane_e,
                                                  plane_u=target_area.plane_u,
                                                  threshold=threshold,
                                                  device=device))
    centers1_tensor = torch.stack(center_of_mass1)
    centers2_tensor = torch.stack(center_of_mass2)
    return torch.mean(torch.norm(centers1_tensor - centers2_tensor, dim=1))


def compute_contour_loss(prediction: torch.Tensor, true_contour: torch.Tensor, target_areas: list()):
    
    assert prediction.shape == true_contour.shape, f"shapes must be equal: {prediction.shape, true_contour.shape}"
    device = prediction.device
    
    # Scale to similar magnitudes      
    loss_scaling = {"SDF": 1e-1, "DICE": 1e-2, "WEIGHT": 0.1}
        
    dice = dice_loss(prediction, true_contour) * loss_scaling["DICE"]
    sdf = sdf_loss(prediction, true_contour) * loss_scaling["SDF"]
    weight = weight_loss(prediction, true_contour, target_areas) * loss_scaling["WEIGHT"]
    ssim = 1 - ssim_loss(prediction.unsqueeze(0), true_contour.unsqueeze(0))
    
    combined_loss = 4.0 * dice + 1.0 * sdf + weight + 0.2 * ssim
    
    return combined_loss, dice.item()
    

def compute_pixels_reflection_axis(orientation: torch.Tensor, incident_ray_direction, target_area):
    """
    Compute the pixels of the intersection of the reflection axis with the target area.
    """
    # Compute ENU-coordinates of intersection (ideal centroid)
    device = orientation.device
    surface_normal = orientation[0:4, 2]
    reflected_ray = raytracing_utils.reflect(incident_ray_direction, surface_normal)
    ideal_center, t = calculate_intersection(orientation[0:4, 3], 
                                             reflected_ray, 
                                             target_area.center, 
                                             target_area.normal_vector)

    # Get pixel indices of ideal centroid
    indices = get_bitmap_indices_from_center_coordinates((256, 256), 
                                                         ideal_center, 
                                                         target_area.center, 
                                                         target_area.plane_e, target_area.plane_u, 
                                                         round=False,
                                                         device=device)
    return indices


def calibrate_on_single_image(scenario_path: pathlib.Path, target_name: str, dir: pathlib.Path(), epochs=300, tower='stj', heliostat="AA39"):

    # Set the device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # === Load and process calibration image
    true_bitmap = processs_image(dir, f'superposed_{heliostat}_raw.png', 'corners.txt', tower=tower)
    
    # === Find contour in ground-truth
    treshold = 0.7
    sharpness = 70
    true_smooth = gaussian_filter_2d(true_bitmap, sigma=2.0)
    true_contour = find_soft_contour_pytorch_vertical(true_bitmap, threshold=treshold, sharpness=sharpness).to(device)
    true_contour = gaussian_filter_2d(true_contour.squeeze(0), sigma=3.0).unsqueeze(0)
    save_bitmap(true_contour, dir/f'contour_gt.png')
    
    # === Load scenario
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario file not found at path: {scenario_path}")
    with h5py.File(scenario_path, "r") as scenario_file:
            loaded_scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file, 
                device=device
            )
    scenario_target_areas = {area.name: area for area in loaded_scenario.target_areas.target_area_list}
    assert target_name in scenario_target_areas, f"{target_name} not found in {scenario_target_areas}"
    
    heliostat_field = loaded_scenario.heliostat_field
    heliostats = heliostat_field.all_heliostat_names
    print(f"Heliostats in scenario: {heliostats}")
    kinematic = heliostat_field.rigid_body_kinematic
    target_areas = [scenario_target_areas[target_name]]
    
    # === Load raytracer
    south_inc_direction = torch.tensor([0.0, 1.0, 0.0, 0.0]) 
    kinematic.aim_points = torch.stack([t.center for t in target_areas])
    heliostat_field.align_surfaces_with_incident_ray_direction(
        incident_ray_direction = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
        device=device
    )
    motor_positions = kinematic.motor_positions
    raytracer = HeliostatRayTracer(scenario=loaded_scenario,
                                   world_size=1, rank=0, batch_size=1, random_seed=42,
                                   bitmap_resolution_e=256, bitmap_resolution_u=256)
    print("Raytracer initialized!")
    
    initial_lr = 0.0001
    
    # === Define learnable parameters and setup optimizer / scheduler
    pred_threshold = torch.nn.Parameter(torch.tensor(treshold, device=device), requires_grad=True)
    kin_params, _, _ = get_rigid_body_kinematic_parameters_from_scenario(kinematic=kinematic)
    
    # Set up Optimizer
    optimizer = torch.optim.Adam(list(kin_params.parameters()) + [pred_threshold], lr=initial_lr)
    
    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="min",
                                                           factor=0.5,
                                                           patience=20,
                                                           threshold=0.0,
                                                           threshold_mode="abs")
    print("Optimizer initialized!")
    
    incident_ray_directions = south_inc_direction.unsqueeze(0).to(device)
    # pred_flux_bitmaps = torch.zeros((len(heliostats), 256, 256), device=device, requires_grad=True)
    log_contours = []
    log_flux = []
    best_criterion = float('inf')
    best_orientation = torch.empty((4,4), device=device)
    
    # === Optimize for n-epochs
    for epoch in range(epochs):
        
        # Align kinematic
        # kinematic.aim_points = torch.stack([t.center for t in target_areas])
        heliostat_field.align_surfaces_with_motor_positions(
            motor_positions=motor_positions,
            device=device
        )
        pred_flux_bitmaps= raytracer.trace_rays_separate(incident_ray_directions=incident_ray_directions,
                                                        target_areas=target_areas,
                                                        device=device)
        
        norm_bitmap = normalize_images(pred_flux_bitmaps).squeeze(0)
        # norm_bitmap = normalize_on_mean(pred_flux_bitmaps.squeeze(0))
        save_bitmap(norm_bitmap, dir/f'prediction_raw.png')
        
        pred_smooth = gaussian_filter_2d(norm_bitmap, sigma=2.0)
        pred_contour = find_soft_contour_pytorch_vertical(pred_smooth, threshold=pred_threshold.sigmoid(), sharpness=sharpness).to(device)
        pred_contour = gaussian_filter_2d(pred_contour.squeeze(0), sigma=3.0).unsqueeze(0)
        save_bitmap(pred_contour.detach(), dir/f'contour_prediction.png')
        
        # Log diff contours every 20 epochs
        if epoch % 10 == 0:
            diff_contour = pred_contour - true_contour
            log_contours.append(diff_contour.squeeze(0).squeeze(0).cpu().detach().numpy())
            diff_flux = norm_bitmap - true_bitmap.to(device)
            log_flux.append(diff_flux.squeeze(0).squeeze(0).cpu().detach().numpy())
        
        # Compute Contour-Image loss
        optimizer.zero_grad()
        loss, criterion = compute_contour_loss(pred_contour, true_contour, target_areas)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if epoch % 5 == 0:
            print(f"epoch {epoch}: {loss.item()} - threshold: {pred_threshold.sigmoid().item()} - learning rate: {optimizer.param_groups[0]['lr']}")
        
        if loss.item() < best_criterion:
            best_loss = loss.item()
            best_orientation = kinematic.orientations[0].detach().clone()
            
    # Save diff plot as gif
    save_bitmaps_as_gif(log_contours, save_under=dir/'contour_diff.gif')
    save_bitmaps_as_gif(log_flux, save_under=dir/'flux_diff.gif')
    
    # === Compute pixel indices of reflection axis and map back to original image
    pixels = compute_pixels_reflection_axis(best_orientation, incident_ray_directions[0], target_areas[0])
    save_bitmap(norm_bitmap.squeeze(0), dir/"prediction_with_center.png", (256, 256), pixels)
    save_bitmap(true_bitmap, dir/"gt_restored_background_removed_with_center.png", (256, 256), pixels)
    original_px = map_restored_flux_pixel_to_composite(
        pixel_coords=pixels,
        flux_scale=3.0,
        tower=tower
    )
    return original_px
    
    
def main():
    
    scenario_dir = pathlib.Path('/dss/dsshome1/05/di38kid/data/scenarios')
    towers = ["cener", "stj"]
    
    heliostats = ["AA39", "AE32", "AF46", "AH30", "AM35", "AO51", "AP30"]
    blockings = ["00", "20", "40"]
    
    for tower in towers:
        for heliostat in heliostats:
            scenario_file = f'{heliostat}.h5'
            
            for blocking in blockings:
                dir = pathlib.Path(f"/dss/dsshome1/05/di38kid/data/max_challenge/{tower.upper()}_{heliostat}_{blocking}")

                center_indices = calibrate_on_single_image(scenario_dir/scenario_file, "solar_tower_juelich_upper", dir=dir, tower=tower, heliostat=heliostat)
                with open(dir / "center_px_indices.txt", "w") as f:
                    f.write(f"[{center_indices[1]}, {center_indices[0]}]")
            
                image = cv2.imread(dir / f"superposed_{heliostat}_raw.png", cv2.IMREAD_GRAYSCALE)
                image_tensor = (torch.tensor(image, dtype=torch.float32))
                
                if tower.upper() == "STJ":
                    target_size = (256, 256)
                else:  # "cener"
                    target_size = (1376, 1024)
                
                save_bitmap(image_tensor, dir / f"superposed_{heliostat}_raw_with_center.png", center_indices=center_indices, target_size=target_size)

if __name__ == '__main__':
    # processs_stj_image(pathlib.Path('/dss/dsshome1/05/di38kid/data/max_challenge'), 'superposed_AA39_raw.png', 'corners.txt')
    main()

