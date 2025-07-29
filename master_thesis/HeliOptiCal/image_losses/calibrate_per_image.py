import torch
import json
import os
import sys
import paint.util.paint_mappings as mappings
from typing import Tuple

from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, animation

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, repo_path)
from HeliOptiCal.calibration_model import CalibrationModel
from HeliOptiCal.utils import my_config_dict
from HeliOptiCal.data_processing.logger import TensorboardLogger
from HeliOptiCal.utils.util import normalize_images, normalize_and_interpolate, get_bitmap_indices_from_center_coordinates
from HeliOptiCal.image_losses.image_loss import find_soft_contour_pytorch_vertical
from HeliOptiCal.utils.util_simulate import gaussian_filter_2d
from HeliOptiCal.utils.util import calculate_intersection

# Add local artist path for raytracing with multiple parallel heliostats.
artist_repo = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, artist_repo)  
from artist.util.scenario import Scenario
from artist.util.utils import get_center_of_mass


"""
def save_training_plots(logs: dict, log_dir: str, calibration_id: int):
    os.makedirs(log_dir, exist_ok=True)
    
    log_alignment_errors = logs['alignment_errors']
    log_true_alignment_errors = logs['true_alignment_errors']
    log_dist_loss = logs['dist_loss']
    log_mse_loss = logs['mse_loss']
        
    epochs = list(range(len(log_alignment_errors)))

    # Plot Losses
    plt.figure()
    plt.plot(epochs, log_mse_loss, label="MSE Loss")
    plt.plot(epochs, log_dist_loss, label="Distance Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Evolution for Calibration ID {calibration_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"loss_curve_{calibration_id}.png"))
    plt.close()

    # Plot Alignment Errors
    plt.figure()
    plt.plot(epochs, log_alignment_errors, label="Alignment Error (Pred)")
    plt.plot(epochs, log_true_alignment_errors, label="Alignment Error (True)")
    plt.xlabel("Epoch")
    plt.ylabel("Alignment Error [mrad]")
    plt.title(f"Alignment Error for Calibration ID {calibration_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"alignment_errors_{calibration_id}.png"))
    plt.close()
"""

def save_training_plots(logs: dict, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)

    all_calibration_ids = logs['calibration_ids']
    all_alignment_errors = logs['alignment_errors']        # List of lists: [n_samples][epochs]
    all_true_alignment_errors = logs['true_alignment_errors']
    all_image_loss = logs['mse_image_loss']
    all_alignment_loss = logs['mse_alignment_loss']
    all_contour_loss = logs['mse_contour_loss']
    all_combined_loss = logs['combined_loss']

    n_samples = len(all_alignment_errors)
    epochs = list(range(len(all_alignment_errors[0])))

    def plot_metric(metric_values, title, ylabel, filename, colors=None):
        plt.figure(figsize=(10, 6))
        for i in range(n_samples):
            label = f"ID {all_calibration_ids[i]}"
            color = colors[i] if colors else None
            plt.plot(epochs, metric_values[i], label=label, color=color)
            
        plt.yscale('log')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, filename))
        plt.close()

    # Plot all metrics
    plot_metric(all_alignment_loss, "MSE Alignment Loss Evolution", "Loss", f"all_mse_alignment_loss.png")
    plot_metric(all_image_loss, "MSE Image Loss Evolution", "Loss", f"all_mse_image_loss.png")
    plot_metric(all_contour_loss, "MSE Contour Loss Evolution", "Loss", f"all_mse_contour_loss.png")
    plot_metric(all_combined_loss, "Combined Loss Evolution", "Loss", f"all_combined_loss.png")
    plot_metric(all_alignment_errors, "Predicted Alignment Error Evolution", "Error [mrad]", f"all_pred_errors.png")
    plot_metric(all_true_alignment_errors, "True Alignment Error Evolution", "Error [mrad]", f"all_true_errors.png")
    

def save_bitmap(bitmap:torch.Tensor, save_under: Path):
    width_in_pixels, height_in_pixels = 512, 512
    dpi = 100  # Can be any value
    width_in_inches = width_in_pixels / dpi
    height_in_inches = height_in_pixels / dpi

    plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
    plt.imshow(bitmap.cpu().detach(), cmap="gray")
    plt.axis("off")  # Hides both x and y axes
    plt.savefig(save_under, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()


def plot_center_cross(center_indices: Tuple[int, int], cross_size: int = 5, cross_color: str = "red", cross_linewidth: float = 1.5):
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
         

def save_bitmap_with_center_cross(bitmap: torch.Tensor, center_indices: Tuple[int, int], save_under: Path):
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


def save_bitmaps_as_gif(bitmaps: [torch.Tensor], save_under: Path, true_center_indices=None, pred_center_indices_list=None):
    fig, ax = plt.subplots()
    im = ax.imshow(bitmaps[0], cmap="gray")
    ax.axis("off")
    
    if true_center_indices is not None:
        plot_center_cross(true_center_indices, cross_color='green')
    
    # Initialize predicted center cross
    pred_hline, = ax.plot([], [], color='red', linewidth=1.5)
    pred_vline, = ax.plot([], [], color='red', linewidth=1.5)
    
    def update(frame):
        im.set_array(bitmaps[frame])
        if pred_center_indices_list is None:
            pred_u, pred_e = pred_center_indices_list[frame]
            
            pred_hline.set_data([pred_e - 5, pred_e + 5], [pred_u, pred_u])
            pred_vline.set_data([pred_e, pred_e], [pred_u - 5, pred_u + 5])
            return [im]
        return [im, pred_hline, pred_vline]
    
    ani = animation.FuncAnimation(fig, update, frames=len(bitmaps), blit=True)
    plt.tight_layout()
    plt.axis("off")  # Hides both x and y axes
    ani.save(save_under, writer='pillow', fps=10, dpi=100)
    
    plt.close(fig)


def test_loss_per_image_sample(config_dict: dict):
    
    directory = Path("/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/image_losses")
    name = Path(f"run_{datetime.now().strftime('%y%m%d%H%M')}_AM35")
    
    general_config = run_config[my_config_dict.run_config_general]
    model = CalibrationModel(run_config=config_dict, name=directory / name)
    device = model.device
    save_dir = model.save_directory
    
    splits = model.datasplitter.splits
    split_df = splits['knn'][(30, 30)]
    
    all_heliostat_ids = model.heliostat_ids
    helio_and_calib_ids = {heliostat_id: split_df.loc[split_df[mappings.HELIOSTAT_ID] == heliostat_id].index.tolist()
                       for heliostat_id in all_heliostat_ids}

    # Get the calibration data as a batch   
    batch = model.dataloader.get_field_batch(helio_and_calib_ids=helio_and_calib_ids, device=device)
    
    logs = {}
    logs['calibration_ids'] = []
    logs['alignment_errors'] = []
    logs['true_alignment_errors'] = []
    logs['mse_image_loss'] = []
    logs['mse_alignment_loss'] = []
    logs['mse_contour_loss'] = []
    logs['combined_loss'] = []
    
    for n_sample, data in enumerate(batch[:10]):
        
        log = dict()
        log_alignment_errors = []
        log_true_alignment_errors = []
        log_mse_loss = []
        log_img_loss = []
        log_combined_loss = []
        log_contour_loss = []
        
        log_contour = []
        log_diff_contour = []
        log_pred_center_indices = []
        
        # Get the required data
        calibration_id = data[my_config_dict.field_sample_calibration_ids][0]
        sun_elevation = data[my_config_dict.field_sample_sun_elevations][0]
        sun_azimuth = data[my_config_dict.field_sample_sun_azimuths][0]
        incident_ray_direction = data[my_config_dict.field_sample_incident_rays][0]
        target_area_name = data[my_config_dict.field_sample_target_names][0]
        shaded_flux_center = data[my_config_dict.field_sample_flux_centers][0]
        true_flux_center = data[my_config_dict.field_sample_ideal_flux_centers][0]
        
        print(f"Running sample {n_sample + 1}/{len(batch)}: {calibration_id}")
        logs['calibration_ids'].append(calibration_id)
        
        # Reload Calibration Model to reset everything
        model = CalibrationModel(run_config=config_dict, name=directory / name / str(calibration_id))
        
        target_area = [model.scenario.get_target_area(target_area_name)]
        true_flux_bitmap = data[my_config_dict.field_sample_flux_images][0]

        true_flux_center_indices = get_bitmap_indices_from_center_coordinates(
            true_flux_bitmap, torch.tensor(true_flux_center, device=device), target_area[0].center, target_area[0].plane_e, target_area[0].plane_u, device
            )

        helio_and_calib_id = {all_heliostat_ids[0]: [calibration_id]}
        data_batch = model.dataloader.get_field_batch(helio_and_calib_ids=helio_and_calib_id, device=device)
            
        # Start Tensorboard Logger
        model.tb_logger = TensorboardLogger(run=str(calibration_id), heliostat_names=all_heliostat_ids, log_dir=directory / name)
        model.tb_logger.set_mode("Train")
        model.tb_logger.set_helio_and_calib_ids(helio_and_calib_id)
        
        # Caclculate the contour for the true image
        threshold = 0.3
        sharpness = 100
        num_interpolations=4
        sigma = 2.5
        
        true_flux_bitmap = gaussian_filter_2d(true_flux_bitmap, sigma=sigma)
        true_flux_bitmap = normalize_and_interpolate(true_flux_bitmap, num_interpolations=num_interpolations).squeeze(0).squeeze(0)
        true_contour = find_soft_contour_pytorch_vertical(true_flux_bitmap, threshold, sharpness).squeeze(0)
        true_contour = normalize_and_interpolate(true_contour, num_interpolations=num_interpolations).squeeze(0).squeeze(0)
        true_contour = gaussian_filter_2d(true_contour, sigma=sigma)
        save_bitmap(true_contour, directory / name / str(calibration_id) / 'True_Contour.png')
        save_bitmap_with_center_cross(true_contour, true_flux_center_indices, directory / name / str(calibration_id) / 'True_Contour_with_Center.png')
        
            
        # Optimize per sample for n_epochs
        n_epochs = 500
        for epoch in range(n_epochs):
                        
            model.train()
            
            orientations, pred_flux_bitmaps = model.forward(data_batch)
            pred_flux_bitmap = pred_flux_bitmaps[0, 0, :, :]
            if epoch % 5 == 0:
                pred_flux_center = get_center_of_mass(pred_flux_bitmap, target_area[0].center, target_area[0].plane_e, target_area[0].plane_u, device=device)
                pred_flux_center_indices = get_bitmap_indices_from_center_coordinates(
                    pred_flux_bitmap, torch.tensor(pred_flux_center, device=device), target_area[0].center, target_area[0].plane_e, target_area[0].plane_u, device
                    )
                log_pred_center_indices.append(pred_flux_center_indices)
            
            alignment_loss, alignment_errors, true_alignment_errors = model.evaluate_model(epoch, orientations, pred_flux_bitmaps, data_batch)
            alignment_loss = alignment_loss* 100000
            
            mse_fnc = torch.nn.MSELoss()
            img_mse = mse_fnc(pred_flux_bitmap, true_flux_bitmap)
            
            pred_flux_bitmap = gaussian_filter_2d(pred_flux_bitmap, sigma=sigma)
            pred_flux_bitmap = normalize_and_interpolate(pred_flux_bitmap, num_interpolations=num_interpolations).squeeze(0).squeeze(0) 
            # threshold = 0.6          
            pred_contour = find_soft_contour_pytorch_vertical(pred_flux_bitmap, threshold, sharpness).squeeze(0)      
            pred_contour = normalize_and_interpolate(pred_contour, num_interpolations=num_interpolations).squeeze(0).squeeze(0)    
            pred_contour = gaussian_filter_2d(pred_contour, sigma=sigma)
            contour_mse = mse_fnc(pred_contour, true_contour)
            
            diff_contour = pred_contour - true_contour
            if epoch % 5 == 0:
                log_contour.append(pred_contour.cpu().detach().numpy())
                log_diff_contour.append(diff_contour.cpu().detach().numpy())
            
            alpha = epoch / 100
            
            if epoch < 200:
                loss = alignment_loss
            elif epoch < 300: 
                loss = contour_mse / 10 * alpha + alignment_loss * (1 - alpha)
            else: 
                loss = contour_mse / 10
            
            #with torch.no_grad():
            #    
            #    _, true_alignment_errors = model.evaluate_model(epoch, orientations, pred_flux_bitmaps, data_batch, test_ideal=True)
            
            
            # dist_loss = model.dist_loss_single_image(true_flux_bitmap, pred_flux_bitmap, epoch)
                
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            
            print(f"[{calibration_id}] [Epoch: {epoch}] [Alignment Loss: {alignment_loss.item()}] [Contour Loss: {contour_mse.item()}] "\
                  f"[Combined Loss: {loss.item()}]" \
                  f"[Error: {alignment_errors.mean().item()}] [True Error: {true_alignment_errors.mean().item()}]")
            
            log_alignment_errors.append(alignment_errors.mean().item())
            log_true_alignment_errors.append(true_alignment_errors.mean().item())
            log_img_loss.append(img_mse.item())
            log_mse_loss.append(alignment_loss.item())
            log_combined_loss.append(loss.item())
            log_contour_loss.append(contour_mse.item())
        
        # save_bitmaps_as_gif(log_contour, directory / name / str(calibration_id) / 'Pred_Contour.gif')
        save_bitmaps_as_gif(log_diff_contour, true_flux_center_indices, log_pred_center_indices, directory / name / str(calibration_id) / 'Diff_Contour.gif', )
        
        logs['alignment_errors'].append(log_alignment_errors)
        logs['true_alignment_errors'].append(log_true_alignment_errors)
        logs['mse_image_loss'].append(log_img_loss)
        logs['mse_alignment_loss'].append(log_mse_loss)
        logs['mse_contour_loss'].append(log_contour_loss)
        logs['combined_loss'].append(log_combined_loss)
        
    print('Generating Plots...')    
    # Change this function, show all results only four plots per log key    
    save_training_plots(logs=logs, log_dir=directory / name / 'plots')


if __name__ == '__main__':
    
    run_config_file = '//dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/image_losses/run_config.json'
    run_config = json.load(open(run_config_file))
    test_loss_per_image_sample(run_config)

    

