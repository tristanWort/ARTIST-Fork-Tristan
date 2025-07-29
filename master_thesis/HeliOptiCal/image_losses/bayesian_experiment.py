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
from HeliOptiCal.image_losses.calibration_model import CalibrationModel
from HeliOptiCal.utils import my_config_dict
from HeliOptiCal.data_processing.logger import TensorboardLogger
from HeliOptiCal.utils.util import normalize_images, normalize_and_interpolate, get_bitmap_indices_from_center_coordinates
from HeliOptiCal.image_losses.image_loss import (find_soft_contour_pytorch_vertical, chamfer_distance_batch_optimized)
from HeliOptiCal.image_losses.image_loss import dist_loss_image_batch
from HeliOptiCal.image_losses.image_loss import sdf_loss, dice_loss
from HeliOptiCal.utils.util_simulate import gaussian_filter_2d
from HeliOptiCal.utils.util import calculate_intersection
from HeliOptiCal.image_losses.calibrate_per_image import save_bitmap, save_bitmap_with_center_cross, save_bitmaps_as_gif

# Add local artist path for raytracing with multiple parallel heliostats.
artist_repo = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, artist_repo)  
from artist.util.scenario import Scenario
from artist.util.utils import get_center_of_mass
from artist.raytracing import raytracing_utils


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
        chd_loss = [log for log in logs['chamfer_distance_loss']]
        combined_loss = [log for log in logs['combined_loss']]
        dice_loss = [log for log in logs['dice_loss']]
        sdf_loss = [log for log in logs['sdf_loss']]
        dist_loss = [log for log in logs['dist_loss']]
        plot_metric(alignment_loss, "MSE Alignment Loss Evolution", "Loss", f"all_mse_alignment_loss.png")
        plot_metric(chd_loss, "Chamfer Distance on Contours Loss Evolution", "Loss", f"all_chd_loss.png")
        plot_metric(contour_loss, "MSE Contour Loss Evolution", "Loss", f"all_mse_contour_loss.png")
        plot_metric(combined_loss, "Combined Loss Evolution", "Loss", f"all_combined_loss.png")
        plot_metric(dice_loss, "DICE Contour Loss Evolution", "Loss", f"all_dice_loss.png")
        plot_metric(sdf_loss, "SDF Evolution", "Loss", f"all_sdf_loss.png")
        plot_metric(dist_loss, "Dist Loss Evolution", "Loss", f"all_dist_loss.png")

    # Plot all metrics
    plot_metric(alignment_errors, "Standard Alignment Error Evolution", "Error [mrad]", f"all_pred_errors.png")
    plot_metric(true_alignment_errors, "True Alignment Error Evolution", "Error [mrad]", f"all_true_errors.png")
    

def run_experiment(run_config, n_epochs, n_samples, 
                   threshold, sharpness, sigma_in, sigma_out, num_interpolations, beta, gamma, epsilon1, epsilon2, omega,
                   tolerance, h_idx=None,  # if set to None, perform experiment for all heliostats in scenario
                   ):
    # === BASIC SETUP
    # Set directory for storing outputs and saving logs.
    directory = Path("/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/image_losses/runs/20Heliostats")
    name = Path(f"run_{datetime.now().strftime('%y%m%d%H%M')}_20Heliostats")  # for unique identification
    
    # Initiate a calibration model
    init_model = CalibrationModel(run_config=run_config, name=directory/name)
    device = init_model.device
    save_dir = init_model.save_directory
    
    # Get Data splits
    splits = init_model.datasplitter.splits
    split_df = splits['knn'][(30, 30)]
    all_heliostat_ids = init_model.heliostat_ids
    helio_and_calib_ids = {heliostat_id: split_df.loc[split_df[mappings.HELIOSTAT_ID] == heliostat_id].index.tolist()
                       for heliostat_id in all_heliostat_ids}
    keep_logger = TensorboardLogger(run='log', heliostat_names=all_heliostat_ids, log_dir=directory / name)

    # Redefine h_idx if None was given 
    if h_idx is None:
        h_idxs = [i for i in range(len(all_heliostat_ids))]
    else:
        h_idxs = [h_idx]
    
    # Get the calibration data as a batch   
    batch = init_model.dataloader.get_field_batch(helio_and_calib_ids=helio_and_calib_ids, device=device)
    
    # Log important metrics and losses
    logs = {}
    logs['calibration_ids'] = []
    logs['alignment_errors'] = []
    logs['true_alignment_errors'] = []
    logs['mse_alignment_loss'] = []
    logs['mse_contour_loss'] = []
    logs['chamfer_distance_loss'] = []
    logs['dist_loss'] = []
    logs['sdf_loss'] = []
    logs['dice_loss'] = []
    logs['combined_loss'] = []

    # save parameters
    param_dict = {'threshold': threshold, 
                  'sharpness': int(sharpness), 
                  'sigma_in': sigma_in, 
                  'sigma_out': sigma_out,
                  'num_interpolations': int(num_interpolations),
                  'beta': beta,
                  'gamma': gamma,
                  "epsilon1": epsilon1,
                  "epsilon2": epsilon2,
                  "omega": omega
                  # 'use_dist': int(use_dist),
                  # 'dist_factor': dist_factor
                  }
    json.dump(param_dict, open(save_dir / 'parameters.json', 'w+'), indent=4)
    
    # Iterate over samples and perform full calibration for each sample (per every image)
    for n_sample, data in enumerate(batch[10:n_samples+10]):
        torch.cuda.empty_cache()
        
        # === PER-SAMPLE SETUP
        # Log statistics for each sample's calibration results
        log = dict()
        n = len(all_heliostat_ids)
        log_alignment_errors = [[] for _ in range(n)]
        log_true_alignment_errors = [[] for _ in range(n)]
        log_mse_loss = []
        log_chd_loss = []
        log_sdf_loss = []
        log_dist_loss = []
        log_dice_loss = []
        log_combined_loss = []
        log_contour_loss = []

        # Log images for contours
        log_contour = [[] for _ in range(n)]
        log_diff_contour = [[] for _ in range(n)]
        log_diff_flux = [[] for _ in range(n)]
        log_pred_center_indices = [[] for _ in range(n)]
        
        # Get the required data from the batch
        calibration_ids = data[my_config_dict.field_sample_calibration_ids]
        sun_elevations = data[my_config_dict.field_sample_sun_elevations]
        sun_azimuths = data[my_config_dict.field_sample_sun_azimuths]
        incident_ray_directions = data[my_config_dict.field_sample_incident_rays]
        target_area_names = data[my_config_dict.field_sample_target_names]
        shaded_flux_centers = data[my_config_dict.field_sample_flux_centers]
        true_flux_centers = data[my_config_dict.field_sample_ideal_flux_centers]
        true_bitmaps = data[my_config_dict.field_sample_flux_images]
        true_bitmaps = normalize_images(true_bitmaps)
        
        print(f"Running sample {n_sample + 1}/{len(batch)}: {calibration_ids}")
        logs['calibration_ids'].append(calibration_ids)
        
        # Reload Calibration Model to reset everything
        model = CalibrationModel(run_config=run_config, name=directory / name / str(n_sample))
        model.tb_logger = keep_logger  # replace logger
        
        # Get the bitmap indices of the true flux centers (for plotting)
        target_areas = [model.scenario.get_target_area(target_area_name) for target_area_name in target_area_names]
        true_flux_center_indices =  [get_bitmap_indices_from_center_coordinates(
            true_bitmaps[i], true_flux_centers[i], target_area.center, target_area.plane_e, target_area.plane_u, device
            ) for i, target_area in enumerate(target_areas)]

        # Prepare databatch for current data sample
        helio_and_calib_ids = {all_heliostat_ids[i]: [calibration_ids[i]] for i in range(len(calibration_ids))}
        data_batch = model.dataloader.get_field_batch(helio_and_calib_ids=helio_and_calib_ids, device=device)
        
        # Set logger
        model.tb_logger.set_mode("Train")
        model.tb_logger.set_helio_and_calib_ids(helio_and_calib_ids)

        
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
            return contours, input_bitmaps
        
        
        def center_of_mass_mse(input_bitmaps1, input_bitmaps2, target_areas, threshold=0):
            
            assert input_bitmaps1.shape == input_bitmaps2.shape
            device = input_bitmaps1.device
            center_of_mass1 = []
            center_of_mass2 = []
            
            for i in range(input_bitmaps1.shape[0]):
                
                target_area = target_areas[i]
                center_of_mass1.append(get_center_of_mass(bitmap=input_bitmaps1[i], 
                                                          target_center=target_area.center,
                                                          plane_e=target_area.plane_e,
                                                          plane_u=target_area.plane_u,
                                                          threshold=threshold,
                                                          device=device))
                center_of_mass2.append(get_center_of_mass(bitmap=input_bitmaps2[i], 
                                                          target_center=target_area.center,
                                                          plane_e=target_area.plane_e,
                                                          plane_u=target_area.plane_u,
                                                          threshold=threshold,
                                                          device=device))
            centers1_tensor = torch.stack(center_of_mass1)
            centers2_tensor = torch.stack(center_of_mass2)
            return torch.mean(torch.abs(centers1_tensor - centers2_tensor))
                
        
        # Find True Contours and save as pngs.
        true_contours, _ = apply_gaussian_find_contours(true_bitmaps, sharpness, threshold, sigma_in, sigma_out, num_interpolations)
        # true_contours = normalize_and_interpolate(true_contours.unsqueeze(0), num_interpolations).squeeze(0)
        for i in h_idxs:
            calibration_id = calibration_ids[i]
            save_bitmap(true_contours[i], directory / name / str(n_sample) / f'{calibration_id}_True_Contour.png')
            save_bitmap_with_center_cross(true_contours[i], true_flux_center_indices[i], directory / name / str(n_sample) / f'{calibration_id}_True_Contour_center.png')
            save_bitmap_with_center_cross(true_bitmaps[i], true_flux_center_indices[i], directory / name / str(n_sample) / f'{calibration_id}_True_Flux_center.png')
            
        # Optimize per sample for n_epochs
        for epoch in range(n_epochs):
            # Set model to train
            model.train()
            # Perform forward pass
            all_orientations, all_pred_bitmaps = model.forward(data_batch)
            
            # Evaluate model 
            alignment_loss, alignment_errors, true_alignment_errors = model.evaluate_model(epoch, all_orientations, all_pred_bitmaps, data_batch)
            
            # divide epochs into four phases
            # first, second, third = n_epochs / 4, n_epochs / 2, n_epochs * (3 / 4)
            
            # Increase sharpness slowly to final sharpness
            # init_sharpness = 10
            # sharpness = init_sharpness
            # if (second <= epoch <= third):
            #    # for interval between second and third, let sharpness increase linearly until final sharpness
            #    sharpness = init_sharpness + (sharpness - init_sharpness) * (epoch - second) / (third - second)                
            # elif epoch > third:
            #     sharpness = sharpness
            
            sharpness = sharpness
            pred_bitmaps = normalize_images(all_pred_bitmaps.squeeze(0))
            pred_contours, _ = apply_gaussian_find_contours(pred_bitmaps, sharpness, threshold, sigma_in, sigma_out, num_interpolations)

            diff_contours = pred_contours - true_contours
            diff_bitmaps = pred_bitmaps - true_bitmaps
            distances = []
            
            # Log centroid coordinates of predictions as well as contours
            for i, area in enumerate(target_areas): 
                pred_bitmap = pred_bitmaps[i]
                # pred_flux_center = get_center_of_mass(pred_bitmap, area.center, area.plane_e, area.plane_u, device=device)
                surface_normal = all_orientations[0, i, 0:4, 2]
                reflected_ray = raytracing_utils.reflect(incident_ray_directions[i], surface_normal)
                ideal_center, t = calculate_intersection(all_orientations[0, i, 0:4, 3], reflected_ray, area.center, area.normal_vector)
                distances.append(t.item())
                
                if epoch % 50 == 0 or epoch == n_epochs - 1:
                    indices = get_bitmap_indices_from_center_coordinates(pred_bitmap, ideal_center, area.center, area.plane_e, area.plane_u, device)
                    log_pred_center_indices[i].append(indices)
                    log_contour[i].append(pred_contours[i].cpu().detach().numpy())
                    log_diff_contour[i].append(diff_contours[i].cpu().detach().numpy())
                    log_diff_flux[i].append(diff_bitmaps[i].cpu().detach().numpy())

            # Save contour prediction for last epoch 
            if epoch == n_epochs - 1:
                for i in h_idxs:
                    calibration_id = calibration_ids[i]
                    save_bitmap(pred_contours[i], directory / name / str(n_sample) / f'{calibration_id}_Pred_Contour.png')
                    save_bitmap_with_center_cross(pred_contours[i], log_pred_center_indices[i][-1], directory/name/str(n_sample)/f'{calibration_id}_Pred_Contour_center.png')
                    save_bitmap_with_center_cross(pred_bitmaps[i], log_pred_center_indices[i][-1], directory/name/str(n_sample)/f'{calibration_id}_Pred_Flux_center.png')
                    
            # Calculate MSE Loss on Flux and Contours
            mse_fnc = torch.nn.MSELoss()
            all_flux_mse  = torch.empty(len(h_idxs), device=device)
            all_contour_mse = torch.empty_like(all_flux_mse)
            all_contour_chd = torch.empty_like(all_flux_mse)
            all_contour_dist = torch.empty_like(all_flux_mse)
            all_alignment_loss = torch.empty_like(all_flux_mse)
            
            for i in h_idxs:
                all_flux_mse[i] = mse_fnc(pred_bitmaps[i], true_bitmaps[i])
                # if bool(use_dist):
                #     all_contour_mse[i] = mse_fnc(pred_contours[i], true_contours[i]) / distances[i]
                # else:
                all_contour_mse[i] = mse_fnc(pred_contours[i], true_contours[i])
                
                all_contour_chd[i] = chamfer_distance_batch_optimized(pred_contours[i].unsqueeze(0), true_contours[i].unsqueeze(0)) / distances[i]
                # all_contour_dist[i] = dist_loss_image_batch(pred_contours[i].unsqueeze(0), true_contours[i].unsqueeze(0))
                all_alignment_loss[i] = alignment_loss[i]
                
            loss_scaling = {"MSE": 1, "CHD": 1e-3, "SDF": 1e-1, "DICE": 1e-2}
            
            sdf = sdf_loss(pred_contours, true_contours) * loss_scaling['SDF']
            dice = dice_loss(pred_contours, true_contours) * loss_scaling['DICE']
            alignment_loss = all_alignment_loss.mean()
            # flux_mse = all_flux_mse.mean() 
            mse_loss = all_contour_mse.mean() * loss_scaling["MSE"]
            chd = all_contour_chd.mean() * loss_scaling["CHD"]
            dist_loss = center_of_mass_mse(pred_contours, true_contours, target_areas)
            
            # Combine losses
            # if epoch < first:  # First epochs only use vectorial alignment loss
            #     loss = alignment_loss
            # elif epoch < second:  # Make gradual transition to image-based loss
            #     alpha1 = (epoch - first) / (second - first)  # 0 at 'first', 1 at 'second'
            #     # loss = alpha1 * ((1-beta) * contour_mse + beta * chd_loss) + (1 - alpha1) * alignment_loss
            #     loss = alpha1 * ((1-beta) * contour_dice_loss + beta * contour_sdf_loss) + (1 - alpha1) * alignment_loss
            # elif epoch < third:
            #     alpha2 = (epoch - second) / (third - second)
            #     # loss = (1-beta) * contour_mse + chd_loss * beta * (1 - alpha2) + alignment_loss * gamma
            #     loss = (1-beta) * contour_dice_loss + contour_sdf_loss * beta  + alignment_loss * gamma
            # else:
            #     loss = (1-beta) * contour_dice_loss + contour_sdf_loss * beta  + alignment_loss * gamma
            #     # loss = contour_dice_loss + alignment_loss * gamma
            
            fine_loss = epsilon1 * dice + (1-epsilon1) * mse_loss
            coarse_loss = epsilon2 * sdf + (1-epsilon2) * chd
            
            warmup = 100
            half = (n_epochs - warmup) / 2 + warmup
            # Make soft transition to contour loss terms in first half
            if epoch < warmup:
                loss = alignment_loss
            elif epoch <= half:
                # alpha goes from 0 at 'warmup' to 1 at 'half' 
                alpha = (epoch - warmup) / (half - warmup) 
                loss = alpha * ((1-beta) * fine_loss + beta * coarse_loss + omega * dist_loss) + (1 - alpha + gamma) * alignment_loss
            else: 
                loss = (1-beta) * fine_loss + beta * coarse_loss + omega * dist_loss + gamma * alignment_loss
            
            # Backward
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            # model.schedulers.step(loss)
            current_lr = model.optimizer.param_groups[0]['lr']
            
            relevant_alignment_erors = [alignment_errors[:, i].item() for i in h_idxs]
            mean_alignment_error = sum(error for error in relevant_alignment_erors) / len(relevant_alignment_erors)
            relevant_true_alignment_erors = [true_alignment_errors[:, i].item() for i in h_idxs]
            mean_true_alignment_error = sum(error for error in relevant_true_alignment_erors) / len(relevant_true_alignment_erors)
            
            # print(f"[Sample: {n_sample+1} / {n_samples}] [Epoch: {epoch} / {n_epochs}]" \
            #       f"[Alignment Loss: {alignment_loss.item(): .6f}] [Contour Loss: {contour_mse.item(): .6f}]" \
            #       f"[CHD Loss: {chd_loss.item(): .6f}] [Loss: {loss.item(): .6f}]" \
            #       f"[Error: {mean_alignment_error: .4f}] [True Error: {mean_true_alignment_error: .4f}]" \
            #       f"[LR: {current_lr: .6f}]")
            
            print(f"[Sample: {n_sample+1} / {n_samples}] [Epoch: {epoch} / {n_epochs}]" \
                  f"[Alignment Loss: {alignment_loss.item(): .6f}] [Fine Loss: {fine_loss.item(): .6f}]" \
                  f"[Coarse Loss: {coarse_loss.item(): .6f}] [Dist Loss: {dist_loss.item(): .6f}] [Loss: {loss.item(): .6f}]" \
                  f"[Error: {mean_alignment_error: .4f}] [True Error: {mean_true_alignment_error: .4f}]" \
                  f"[LR: {current_lr: .6f}]")
            
            # Log the metrics and losses
            for i in range(len(calibration_ids)):
                log_alignment_errors[i].append(alignment_errors[0, i].item())
                log_true_alignment_errors[i].append(true_alignment_errors[0, i].item())
                
            log_mse_loss.append(alignment_loss.item())
            log_contour_loss.append(mse_loss.item())
            log_chd_loss.append(chd.item())
            log_sdf_loss.append(sdf.item())
            log_dist_loss.append(dist_loss.item())
            log_dice_loss.append(dice.item())
            log_combined_loss.append(loss.item())
            
        logs['alignment_errors'].append(log_alignment_errors)
        logs['true_alignment_errors'].append(log_true_alignment_errors)
        logs['chamfer_distance_loss'].append(log_chd_loss)
        logs['sdf_loss'].append(log_sdf_loss)
        logs['dist_loss'].append(log_dist_loss)
        logs['dice_loss'].append(log_dice_loss)
        logs['mse_alignment_loss'].append(log_mse_loss)
        logs['mse_contour_loss'].append(log_contour_loss)
        logs['combined_loss'].append(log_combined_loss)
        
        for i in h_idxs:
            cal_id = calibration_ids[i]
            log_diff_c = log_diff_contour[i]
            save_bitmaps_as_gif(log_diff_c, true_flux_center_indices[i], log_pred_center_indices[i], directory/name/str(n_sample)/f'{cal_id}_Diff_Contour.gif')
            log_diff_f = log_diff_flux[i]
            # save_bitmaps_as_gif(log_diff_f, true_flux_center_indices[i], log_pred_center_indices[i],  directory/name/str(n_sample)/f'{cal_id}_Diff_Flux.gif')
            
        # Count number of final alignment errors that are over threshold [mrad]
        # last_errors = [helio_errors[-1] for sample in logs['true_alignment_errors'] for helio_errors in sample]
        # num_above_threshold = sum(1 for err in last_errors if err > 0.5)
        # Compute MSE of all final alignment errors
        # mse_last_errors = sum(err**2 for err in last_errors) / len(last_errors)
        
        last_errors = [sample[i][-1] for sample in logs['true_alignment_errors'] for i in h_idxs]
        # last_errors = [sample[h_idx][-1] for sample in logs['true_alignment_errors'] for ]
        mse_last_errors = sum(err**2 for err in last_errors) / len(last_errors)
        # Stop exploration
        if mse_last_errors > tolerance:
            break
        
    keep_logger.close()
    print('Done with all samples. Generating Plots...')      
    for i in h_idxs:
        heliostat = all_heliostat_ids[i]
        save_training_plots_multiple_heliostats(logs=logs, helio_idx=i, log_dir=directory / name / f'plots/{heliostat}')

    return mse_last_errors
        