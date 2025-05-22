"""
A function for executing experiments for contour-based alignment.
Can be executed from the Bayesian optimizer to find the best parameters.
"""

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
from HeliOptiCal.image_losses.calibrate_per_image import save_bitmap, save_bitmap_with_center_cross, save_bitmaps_as_gif

# Add local artist path for raytracing with multiple parallel heliostats.
artist_repo = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, artist_repo)  
from artist.util.scenario import Scenario
from artist.util.utils import get_center_of_mass


def save_training_plots_multiple_heliostats(logs: dict, helio_idx: int, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    
    calibration_ids = [log[helio_idx] for log in logs['calibration_ids']]
    alignment_errors = [log[helio_idx] for log in logs['alignment_errors']]
    true_alignment_errors = [log[helio_idx] for log in logs['true_alignment_errors']] 
    
    n_samples = len(alignment_errors)
    epochs = list(range(len(alignment_errors[0])))
    
    def plot_metric(metric_values, title, ylabel, filename, colors=None):
        plt.figure(figsize=(10, 6))
        for i in range(n_samples):
            label = f"ID {calibration_ids[i]}"
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
  
    # all_image_loss = logs['mse_image_loss']
    if helio_idx == 0:
        alignment_loss = [log for log in logs['mse_alignment_loss']]
        contour_loss = [log for log in logs['mse_contour_loss']]
        combined_loss = [log for log in logs['combined_loss']]
        plot_metric(alignment_loss, "MSE Alignment Loss Evolution", "Loss", f"all_mse_alignment_loss.png")
        # plot_metric(image_loss, "MSE Image Loss Evolution", "Loss", f"all_mse_image_loss.png")
        plot_metric(contour_loss, "MSE Contour Loss Evolution", "Loss", f"all_mse_contour_loss.png")
        plot_metric(combined_loss, "Combined Loss Evolution", "Loss", f"all_combined_loss.png")

    # Plot all metrics
    plot_metric(alignment_errors, "Standard Alignment Error Evolution", "Error [mrad]", f"all_pred_errors.png")
    plot_metric(true_alignment_errors, "True Alignment Error Evolution", "Error [mrad]", f"all_true_errors.png")
    

def run_experiment(run_config, n_epochs, n_samples, threshold, final_sharpness, num_interpolations, sigma_out, tolerance=10):
    
    # Set directory for storing outputs and saving logs.
    directory = Path("/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/image_losses")
    name = Path(f"run_{datetime.now().strftime('%y%m%d%H%M')}_AA39")  # for unique identification
    
    # Initiate a calibration model
    init_model = CalibrationModel(run_config=run_config, name=directory / name)
    device = init_model.device
    save_dir = init_model.save_directory
    
    splits = init_model.datasplitter.splits
    split_df = splits['knn'][(30, 30)]
    
    all_heliostat_ids = init_model.heliostat_ids
    helio_and_calib_ids = {heliostat_id: split_df.loc[split_df[mappings.HELIOSTAT_ID] == heliostat_id].index.tolist()
                       for heliostat_id in all_heliostat_ids}
    keep_logger = TensorboardLogger(run='log', heliostat_names=all_heliostat_ids, log_dir=directory / name)

    # Get the calibration data as a batch   
    batch = init_model.dataloader.get_field_batch(helio_and_calib_ids=helio_and_calib_ids, device=device)
    
    # Log important metrics and losses
    logs = {}
    
    logs['calibration_ids'] = []
    logs['alignment_errors'] = []
    logs['true_alignment_errors'] = []
    logs['mse_image_loss'] = []
    logs['mse_alignment_loss'] = []
    logs['mse_contour_loss'] = []
    logs['combined_loss'] = []
    
    # save parameters
    # param_dict = {'threshold': threshold, 'sharpness': sharpness, 'sigma_in:': sigma_in, 'sigma_out': sigma_out, 'num_interpolations': int(num_interpolations)}
    param_dict = {'threshold': threshold, 'final_sharpness': final_sharpness, 'num_interpolations': int(num_interpolations), 'sigma_out': sigma_out}
    json.dump(param_dict, open(save_dir / 'parameters.json', 'w+'), indent=4)
    
    # Iterate over samples and perform full calibration for each sample (per every image)
    for n_sample, data in enumerate(batch[:n_samples]):
        torch.cuda.empty_cache()
        
        # Log statistics for each sample's calibration results
        log = dict()
        n = len(all_heliostat_ids)
        log_alignment_errors = [[] for _ in range(n)]
        log_true_alignment_errors = [[] for _ in range(n)]
        log_mse_loss = []
        log_img_loss = []
        log_combined_loss = []
        log_contour_loss = []

        # Log images for contours
        log_contour = [[] for _ in range(n)]
        log_diff_contour = [[] for _ in range(n)]
        log_pred_center_indices = [[] for _ in range(n)]
        
        # Get the required data from the batch
        calibration_ids = data[my_config_dict.field_sample_calibration_ids]
        sun_elevations = data[my_config_dict.field_sample_sun_elevations]
        sun_azimuths = data[my_config_dict.field_sample_sun_azimuths]
        incident_ray_directions = data[my_config_dict.field_sample_incident_rays]
        target_area_names = data[my_config_dict.field_sample_target_names]
        shaded_flux_centers = data[my_config_dict.field_sample_flux_centers]
        true_flux_centers = data[my_config_dict.field_sample_ideal_flux_centers]
        true_flux_bitmaps = data[my_config_dict.field_sample_flux_images]

        print(f"Running sample {n_sample + 1}/{len(batch)}: {calibration_ids}")
        logs['calibration_ids'].append(calibration_ids)
        
        # Reload Calibration Model to reset everything
        model = CalibrationModel(run_config=run_config, name=directory / name / str(n_sample))
        model.tb_logger = keep_logger  # replace logger
        
        # Get the bitmap indices of the true flux centers (for plotting)
        target_areas = [model.scenario.get_target_area(target_area_name) for target_area_name in target_area_names]
        true_flux_bitmaps = normalize_and_interpolate(true_flux_bitmaps, num_interpolations).squeeze(0)
        true_flux_center_indices =  [get_bitmap_indices_from_center_coordinates(
            true_flux_bitmaps[i], true_flux_centers[i], target_area.center, target_area.plane_e, target_area.plane_u, device
            ) for i, target_area in enumerate(target_areas)]

        # Prepare databatch for current data sample
        helio_and_calib_ids = {all_heliostat_ids[i]: [calibration_ids[i]] for i in range(len(calibration_ids))}
        data_batch = model.dataloader.get_field_batch(helio_and_calib_ids=helio_and_calib_ids, device=device)
        
        # Set logger
        model.tb_logger.set_mode("Train")
        model.tb_logger.set_helio_and_calib_ids(helio_and_calib_ids)
        
        
        def apply_gaussian_find_contours(input_bitmaps, sigma_out=2.5, sharpness=20, threshold=0.1):
            """
            Function for applying gaussian blurring to input bitmap and returning the upper contour.
            """
            # Apply gauss on input
            gauss_contours = torch.empty_like(input_bitmaps)
            for i in range(input_bitmaps.shape[0]):
                # gauss_bitmap = gaussian_filter_2d(input_bitmaps[i], sigma=sigma_in)
                contour = find_soft_contour_pytorch_vertical(input_bitmaps[i], threshold=threshold, sharpness=sharpness)
                gauss_contours[i] = gaussian_filter_2d(contour.squeeze(0), sigma=sigma_out)
            gauss_contours = normalize_images(gauss_contours)
            return gauss_contours
        
        # Find True Contours and save as pngs.
        true_contours = apply_gaussian_find_contours(true_flux_bitmaps, sigma_out, final_sharpness, threshold)
        # true_contours = normalize_and_interpolate(true_contours.unsqueeze(0), num_interpolations).squeeze(0)
        for i in range(true_contours.shape[0]):
            calibration_id = calibration_ids[i]
            save_bitmap(true_contours[i], directory / name / str(n_sample) / f'{calibration_id}_True_Contour.png')
            save_bitmap_with_center_cross(true_contours[i], true_flux_center_indices[i], directory / name / str(n_sample) / f'{calibration_id}_True_Contour_center.png')
        
        sharpness = 10
        # Optimize per sample for n_epochs
        for epoch in range(n_epochs):
            # Set model to train
            model.train()
            # Perform forward pass
            all_orientations, all_pred_flux_bitmaps = model.forward(data_batch)
            
            # Evaluate model 
            alignment_loss, alignment_errors, true_alignment_errors = model.evaluate_model(epoch, all_orientations, all_pred_flux_bitmaps, data_batch)

            # Find Contour of predicted bitmap
            # pred_flux_bitmaps = normalize_images(all_pred_flux_bitmaps[0, :, :, :])  # Only first sample is used
            if epoch > 200:
                sharpness = 10 + (epoch - 200) / 400 * (final_sharpness - 10) 

            sharpness = min(sharpness, final_sharpness)

            pred_flux_bitmaps = normalize_and_interpolate(all_pred_flux_bitmaps, num_interpolations).squeeze(0)
            pred_contours = apply_gaussian_find_contours(pred_flux_bitmaps, sigma_out, sharpness, threshold)
            # pred_contours = normalize_and_interpolate(pred_contours.unsqueeze(0), num_interpolations).squeeze(0)
            diff_contours = pred_contours - true_contours
            
            if epoch % 40 == 0 or epoch == n_epochs - 1:
                # Log centroid coordinates of predictions as well as contours
                for i, area in enumerate(target_areas): 
                    pred_flux_bitmap = pred_flux_bitmaps[i]
                    pred_flux_center = get_center_of_mass(pred_flux_bitmap, area.center, area.plane_e, area.plane_u, device=device)
                    indices = get_bitmap_indices_from_center_coordinates(pred_flux_bitmap, torch.tensor(pred_flux_center, device=device), area.center, area.plane_e, area.plane_u, device)
                    log_pred_center_indices[i].append(indices)
                    log_contour[i].append(pred_contours[i].cpu().detach().numpy())
                    log_diff_contour[i].append(diff_contours[i].cpu().detach().numpy())

            # Save contour prediction for last epoch 
            if epoch == n_epochs - 1:
                for i in range(pred_contours.shape[0]):
                    calibration_id = calibration_ids[i]
                    save_bitmap(pred_contours[i], directory / name / str(n_sample) / f'{calibration_id}_Pred_Contour.png')
                    save_bitmap_with_center_cross(pred_contours[i], log_pred_center_indices[i][-1], directory/name/str(n_sample)/f'{calibration_id}_Pred_Contour_center.png')
                
            # Calculate Contour MSE Loss
            mse_fnc = torch.nn.MSELoss()
            contour_mse = mse_fnc(pred_contours, true_contours)

            # Combine losses
            alignment_loss *= 1e6
            if epoch < 200:
                loss = alignment_loss
            elif epoch < 600: 
                alpha = (epoch - 200) / 400
                loss = contour_mse * alpha + alignment_loss * (1 - alpha)
            else: 
                loss = contour_mse + alignment_loss / 10
                
            # Backward
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            model.schedulers.step(loss)
            current_lr = model.optimizer.param_groups[0]['lr']
            
            print(f"[Sample: {n_sample+1} / {n_samples}] [Epoch: {epoch} / {n_epochs}]" \
                  f"[Alignment Loss: {alignment_loss.item(): .6f}] [Contour Loss: {contour_mse.item(): .6f}] [Loss: {loss.item(): .6f}]" \
                  f"[Error: {alignment_errors.mean().item(): .4f}] [True Error: {true_alignment_errors.mean().item(): .4f}]" \
                  f"[LR: {current_lr: .10f}]")
            
            # Log the metrics and losses
            for i in range(len(calibration_ids)):
                log_alignment_errors[i].append(alignment_errors[0, i].item())
                log_true_alignment_errors[i].append(true_alignment_errors[0, i].item())
                
            log_mse_loss.append(alignment_loss.item())
            log_combined_loss.append(loss.item())
            log_contour_loss.append(contour_mse.item())
            
        logs['alignment_errors'].append(log_alignment_errors)
        logs['true_alignment_errors'].append(log_true_alignment_errors)
        # logs['mse_image_loss'].append(log_img_loss)
        logs['mse_alignment_loss'].append(log_mse_loss)
        logs['mse_contour_loss'].append(log_contour_loss)
        logs['combined_loss'].append(log_combined_loss)
        
        for i, log_diff_c in enumerate(log_diff_contour):
            cal_id = calibration_ids[i]
            save_bitmaps_as_gif(log_diff_c, true_flux_center_indices[i], log_pred_center_indices[i], directory/name/str(n_sample)/f'{cal_id}_Diff_Contour.gif', )
        
        # Count number of final alignment errors that are over threshold [mrad]
        last_errors = [helio_errors[-1] for sample in logs['true_alignment_errors'] for helio_errors in sample]
        num_above_threshold = sum(1 for err in last_errors if err > 0.5)
        # Compute MSE of all final alignment errors
        mse_last_errors = sum(err**2 for err in last_errors) / len(last_errors)
        
        # Stop exploration
        if mse_last_errors > tolerance:
            break
        
    keep_logger.close()
    print('Done with all samples. Generating Plots...')      
    for i, heliostat in enumerate(all_heliostat_ids):
        save_training_plots_multiple_heliostats(logs=logs, helio_idx=i, log_dir=directory / name / f'plots/{heliostat}')

    return mse_last_errors, num_above_threshold
        