"""
This module defines the `CalibrationModel` class, which represents a heliostat calibration model.
It includes methods for initializing the model, configuring optimizers and schedulers, loading scenarios,
and performing calibration tasks. The model is designed to work with the ARTIST framework for ray tracing
and heliostat field simulation.

Dependencies:
- PyTorch for deep learning and optimization
- ARTIST framework for ray tracing and scenario management
- Various utility modules for data processing, logging, and plotting
"""
import torch
import torch.nn as nn
import logging
import sys
import os
import re
import json
import scipy
import copy
from datetime import datetime
from typing import Union, Literal, List, Tuple, Dict, Optional
from pathlib import Path


# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path)
from artist.util.scenario import Scenario
from artist.raytracing import raytracing_utils
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.utils import get_center_of_mass

# Import local modules
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, repo_path)
import HeliOptiCal.utils.my_config_dict as my_config_dict
from HeliOptiCal.data_processing.calibration_datasplitter import CalibrationDataSplitter
from HeliOptiCal.data_processing.calibration_dataloader import CalibrationDataLoader
from HeliOptiCal.data_processing.logger import TensorboardLogger
from HeliOptiCal.utils.util import (
    get_rigid_body_kinematic_parameters_from_scenario, 
    count_parameters, 
    check_for_nan_grad, 
    create_parameter_groups, 
    find_soft_contour_vertical,
    normalize_and_interpolate, 
    normalize_images,
    calculate_intersection)
from HeliOptiCal.plot_results.plot_errors_distributions import analyze_heliostat_field
from HeliOptiCal.image_losses.image_loss import find_soft_contour_pytorch_vertical, sdf_loss, chamfer_distance_batch_optimized, dice_loss
from HeliOptiCal.utils.util_simulate import gaussian_filter_2d

# Configure logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')


class CalibrationModel(nn.Module):
    """
    A class to represent a Heliostat Calibration Model.

    This class is designed to calibrate heliostat fields using ray tracing and optimization techniques.
    It supports various configurations for optimizers, schedulers, and data loaders.

    Attributes:
        run_config (Dict): Configuration dictionary for the model.
        name (str): Name of the model instance.
        save_directory (Path): Directory to save model outputs.
        tb_logger (TensorboardLogger): Logger for TensorBoard.
        val_error_history (List): History of validation errors.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        scenario (Scenario): ARTIST scenario object.
        target_areas (Dict): Target areas in the scenario.
        heliostat_ids (List): List of heliostat IDs in the field.
        use_raytracing (bool): Whether to use ray tracing.
        raytracer (HeliostatRayTracer): Ray tracer instance.
        learnable_params_dict (Dict): Dictionary of learnable parameters.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        schedulers (torch.optim.lr_scheduler): Learning rate schedulers.
        datasplitter (CalibrationDataSplitter): Data splitter for calibration datasets.
        dataloader (CalibrationDataLoader): Data loader for calibration datasets.
    """
    def __init__(self, run_config: Dict[str, Dict]) -> None:
        """
        Initialize a Heliostat Calibration Model.

        Parameters:
            run_config (Dict): Configuration dictionary for the model.
        """
        super(CalibrationModel, self).__init__()
        
        # Perform general model configurations
        self.run_config = run_config  # save for later usage
        general_config = run_config[my_config_dict.run_config_general]

        # Set the model name and save directory
        self.name = f"run_{datetime.now().strftime('%y%m%d%H%M')}_{general_config[my_config_dict.general_name]}"
        log.info(f"Initializing a Heliostat Calibration Model: {self.name}")
        self.save_directory = Path(general_config[my_config_dict.general_save_to_directory]) / self.name

        # Initialize attributes
        self.tb_logger = None  # later set logger for present run
        self.val_error_history = []  # log average mean validation error
        self.device = general_config[my_config_dict.general_device]
    
        # Load the scenario and target areas
        self.scenario, self.target_areas = self.load_scenario(general_config[my_config_dict.general_scenario_path])
        self.heliostat_ids = self.scenario.heliostat_field.all_heliostat_names

        # Initialize Raytracer
        self.use_raytracing = bool(general_config[my_config_dict.general_use_raytracing])
        self.raytracer = self.init_raytraycer(self.scenario)
        log.info("Done with Scenario loading and HeliostatRayTracer setup.")
        
        # Configure the model
        model_config = run_config[my_config_dict.run_config_model]
        self.learnable_params_dict, parameter_count, element_count = (get_rigid_body_kinematic_parameters_from_scenario(
                kinematic=self.scenario.heliostat_field.rigid_body_kinematic))
        log.info(f"Found number of parameters and elements: {parameter_count}, {element_count}")

        # Configure the optimizer and schedulers
        self.optimizer = self.configure_optimizer(model_config)
        self.schedulers = self.configure_schedulers(model_config)
        log.info("Done with optimizer and scheduler setup.")

        # Configure the data loader
        self.datasplitter, self.dataloader = self.configure_dataloader(run_config[my_config_dict.run_config_dataset])
        log.info("Done with data loading data splits.")

        # Save the run configuration to a JSON file
        json.dump(run_config, open(self.save_directory / 'run_config.json', 'w+'), indent=4)
        
    def load_scenario(self, scenario_path: Union[str, Path]) -> Tuple[Scenario, Dict[str, object]]:
        """
        Load and return an ARTIST Scenario.

        Parameters:
            scenario_path (Union[str, Path]): Path to the scenario file.

        Returns:
            Tuple[Scenario, Dict]: Loaded scenario and target areas.
        """
        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Scenario file not found at path: {scenario_path}")

        import h5py
        # Load the scenario from the h5-file.
        with h5py.File(scenario_path, "r") as scenario_file:
            loaded_scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file, 
                device=self.device
            )

        # Add random errors to the kinematic model if configured
        if bool(self.run_config[my_config_dict.run_config_general][my_config_dict.general_introduce_random_errors]):
            from HeliOptiCal.utils.util_errors import add_random_errors_to_kinematic
            error_config = self.run_config[my_config_dict.run_config_initial_errors]
            seed = self.run_config[my_config_dict.run_config_general][my_config_dict.general_random_seed]            
            error_kinematic = add_random_errors_to_kinematic(error_config, loaded_scenario, self.save_directory/'errors',
                                           seed=seed, device=self.device)
            loaded_scenario.heliostat_field.rigid_body_kinematic = error_kinematic
            log.info("Added initial errors to Heliostat field kinematic.")

        target_areas = {area.name: area for area in loaded_scenario.target_areas.target_area_list}
        return loaded_scenario, target_areas
    
    def init_raytraycer(self, scenario: Scenario, world_size=1, rank=0, batch_size=1, random_seed=42, bitmap_resolution=(256, 256)) -> HeliostatRayTracer:
        """
        Initialize and return a HeliostatRayTracer.

        Parameters:
            scenario (Scenario): ARTIST scenario object.
            world_size (int): Number of distributed processes (default: 1).
            rank (int): Rank of the current process (default: 0).
            batch_size (int): Batch size for ray tracing (default: 1).
            random_seed (int): Random seed for reproducibility (default: 42).
            bitmap_resolution (Tuple[int, int]): Resolution of the bitmap (default: (256, 256)).

        Returns:
            HeliostatRayTracer: Initialized ray tracer.
        """
        # Get the loss type configurations 
        loss_config = self.run_config[my_config_dict.run_config_general][my_config_dict.general_loss_config]
        loss_basis = loss_config[my_config_dict.general_loss_basis].lower()
        
        # Perform sanity checks for ray tracing and loss type
        if self.use_raytracing and ('orientation' in loss_basis):
            log.warning(f"use_raytracing is set to 'True' although your loss only considers Heliostat orientations."\
                "Using the Raytracer results in a high demand on computing and increased runtime. Make sure that this is what you want.")
        elif not self.use_raytracing and not ('orientation' in loss_basis):
            raise TypeError(f"loss_basis is configured to {loss_basis} which requires flux outputs from Raytracing. Set use_raytracing to True.")
            
        # Align heliostat field surfaces with the incident ray direction
        heliostat_field = scenario.heliostat_field
        heliostat_field.align_surfaces_with_incident_ray_direction(
            # initial alignment (direction does not matter)
            incident_ray_direction = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device),
            device=self.device
            )

        # Initialize the ray tracer
        raytracer = HeliostatRayTracer(scenario=scenario,
                                       world_size=world_size,
                                       rank=rank,
                                       batch_size=batch_size,
                                       random_seed=random_seed,
                                       bitmap_resolution_e=bitmap_resolution[0],
                                       bitmap_resolution_u=bitmap_resolution[1])
        return raytracer
    
    def configure_optimizer(self, model_config: Dict) -> torch.optim.Optimizer:
        """
        Configure and return the optimizer based on the model configuration.

        Parameters:
            model_config (Dict): Configuration dictionary for the model.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        optimizer_type = model_config[my_config_dict.model_optimizer]
        params = self.learnable_params_dict.parameters()
        initial_lr = model_config[my_config_dict.model_initial_lr]
        weight_decay = model_config[my_config_dict.model_weight_decay]
        
        if optimizer_type == 'SGD':
            sgd = model_config[my_config_dict.model_optimizer_sgd]
            optimizer = torch.optim.SGD(
                params,
                lr=initial_lr,
                momentum=sgd[my_config_dict.model_sgd_momentum],
                nesterov=bool(sgd[my_config_dict.model_sgd_nesterov])
                )
            
        elif optimizer_type == 'RMSprop':
            rmsprop = model_config[my_config_dict.model_optimizer_rmsprop]
            optimizer = torch.optim.RMSprop(
                params,
                lr=initial_lr,
                weight_decay=weight_decay,
                alpha=rmsprop[my_config_dict.model_rmsprop_alpha],
                momentum=rmsprop[my_config_dict.model_rmsprop_momentum],
                centered=bool(rmsprop[my_config_dict.model_rmsprop_centered])
                )

        elif optimizer_type == 'Adam':
            adam = model_config[my_config_dict.model_optimizer_adam]
            optimizer = torch.optim.Adam(
                params,
                lr=initial_lr,
                weight_decay=weight_decay,
                betas=tuple(adam[my_config_dict.model_adam_betas])
                )

        elif optimizer_type == 'AdamW':
            adamw = model_config[my_config_dict.model_optimizer_adamw]
            optimizer = torch.optim.AdamW(
                params,
                lr=initial_lr,
                weight_decay=weight_decay,
                betas=tuple(adamw[my_config_dict.model_adamw_betas])
                )   

        elif optimizer_type == 'Adadelta':
            adadelta = model_config[my_config_dict.model_optimizer_adadelta]
            optimizer = torch.optim.Adadelta(
                params,
                lr=initial_lr,
                weight_decay=weight_decay,
                rho=adadelta[my_config_dict.model_adadelta_rho]
                )

        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        return optimizer
    
    def configure_schedulers(self, model_config: Dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Configure and return learning rate schedulers based on the model configuration.

        Parameters:
            model_config (Dict): Configuration dictionary for the model.

        Returns:
            torch.optim.lr_scheduler: Configured learning rate scheduler(s).
        """
        optimizer = self.optimizer
        initial_lr = model_config[my_config_dict.model_initial_lr]
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"Expected optimizer to be an instance of "
                            f"torch.optim.Optimizer, but got {type(optimizer)}.")

        scheduler_sequence = model_config[my_config_dict.model_scheduler_sequence]
        epochs_sequence = model_config[my_config_dict.model_epochs_sequence]

        if len(scheduler_sequence) == 0:
            return None  # No scheduler configured

        assert len(scheduler_sequence) == len(epochs_sequence), \
            "Length of scheduler_sequence and epochs_sequence must be equal."
            
        # Single scheduler case
        if len(scheduler_sequence) == 1:
            scheduler_type = scheduler_sequence[0]
            num_epochs = epochs_sequence[0]

            if scheduler_type == 'ConstantLR':
                cfg = model_config[my_config_dict.model_constantlr]
                scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=cfg[my_config_dict.model_constantlr_lr_factor],
                    total_iters=num_epochs
                )

            elif scheduler_type == 'CyclicLR':
                cfg = model_config[my_config_dict.model_cycliclr]
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=cfg[my_config_dict.model_cycliclr_base_lr],
                    max_lr=cfg[my_config_dict.model_cycliclr_max_lr],
                    step_size_up=cfg[my_config_dict.model_cycliclr_step_size_up],
                    mode=cfg[my_config_dict.model_cycliclr_mode],
                    gamma=cfg[my_config_dict.model_cycliclr_gamma],
                    cycle_momentum=False
                )
                
            elif scheduler_type == "RampLR":
                cfg = model_config[my_config_dict.model_ramplr]
                final_lr = cfg[my_config_dict.model_cycliclr_max_lr]
                step_up = cfg[my_config_dict.model_cycliclr_step_size_up]
                def lr_lambda(epoch):
                    if epoch < step_up: 
                        # Linear increase until final_lr
                        return (final_lr - initial_lr) / initial_lr * (epoch / step_up) + 1.0
                    else:
                        return final_lr / initial_lr
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

            elif scheduler_type == 'ReduceLROnPlateau':
                cfg = model_config[my_config_dict.model_reduce_on_plateau]
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=cfg[my_config_dict.model_reduce_on_plateau_factor],
                    patience=cfg[my_config_dict.model_reduce_on_plateau_patience],
                    threshold=cfg[my_config_dict.model_reduce_on_plateau_treshold],
                    threshold_mode="abs",
                    min_lr=cfg[my_config_dict.model_reduce_on_plateau_min_lr],
                    verbose=False
                )

            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

            return scheduler

        # Multiple schedulers: use SequentialLR
        schedulers = []
        milestones = []
        epoch_cumsum = 0

        for i, (scheduler_type, num_epochs) in enumerate(zip(scheduler_sequence, epochs_sequence)):
            if scheduler_type == 'ConstantLR':
                cfg = model_config[my_config_dict.model_constantlr]
                scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=cfg[my_config_dict.model_constantlr_lr_factor],
                    total_iters=num_epochs
                )

            elif scheduler_type == 'CyclicLR':
                cfg = model_config[my_config_dict.model_cycliclr]
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=cfg[my_config_dict.model_cycliclr_base_lr],
                    max_lr=cfg[my_config_dict.model_cycliclr_max_lr],
                    step_size_up=num_epochs,
                    mode=cfg[my_config_dict.model_cycliclr_mode],
                    gamma=cfg[my_config_dict.model_cycliclr_gamma],
                    cycle_momentum=False
                )

            elif scheduler_type == 'ReduceLROnPlateau':
                raise ValueError("ReduceLROnPlateau cannot be used within SequentialLR; "
                                "it requires validation metrics and doesn't follow fixed epochs.")

            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

            schedulers.append(scheduler)
            epoch_cumsum += num_epochs
            if i < len(scheduler_sequence) - 1:
                milestones.append(epoch_cumsum)

        return torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                     schedulers=schedulers,
                                                     milestones=milestones)
        
    def configure_dataloader(self, dataset_config: Dict) -> Tuple[CalibrationDataSplitter, CalibrationDataLoader]:
        """
        Configure and return the data splitter and data loader.

        Parameters:
            dataset_config (Dict): Configuration dictionary for the dataset.

        Returns:
            Tuple[CalibrationDataSplitter, CalibrationDataLoader]: Configured data splitter and data loader.
        """
        # Perform datasplits
        data_splitter = CalibrationDataSplitter(
            metadata_path=dataset_config[my_config_dict.dataset_metadata_path],
            output_directory=self.save_directory / 'splits',
        )  
        data_splitter.perform_splits(
            training_sizes=dataset_config[my_config_dict.dataset_training_sizes],
            validation_sizes=dataset_config[my_config_dict.dataset_validation_sizes],
            split_types=dataset_config[my_config_dict.dataset_split_types],
            save_splits_plots=True
        )

        # Configure dataloader for training
        calibration_data_loader = CalibrationDataLoader(
            data_directory=dataset_config[my_config_dict.dataset_training_data_directory],
            heliostats_to_load=self.heliostat_ids,
            power_plant_position=self.scenario.power_plant_position,
            load_flux_images=bool(dataset_config[my_config_dict.dataset_load_flux_images]),
            preload_flux_images=bool(dataset_config[my_config_dict.dataset_preload_flux_images]),
            is_simulated_data=bool(dataset_config[my_config_dict.dataset_training_data_was_simulated]),
            properties_file_ends_with=dataset_config[my_config_dict.dataset_properties_suffix],
            flux_file_ends_with=dataset_config[my_config_dict.dataset_flux_suffix],
            device=dataset_config[my_config_dict.dataset_device]
        )
        return data_splitter, calibration_data_loader

    def calibrate(self, device: Optional[Union[torch.device, str]] = None) -> None:
        """
        Run end‑to‑end calibration across all configured dataset splits.

        The routine iterates over each split (train/validation/test) produced by
        `CalibrationDataSplitter`, performs phased/frozen training as configured,
        logs metrics and artifacts to TensorBoard, tracks the best validation state,
        and finally evaluates on the test set.

        Parameters
        ----------
        device : torch.device or str, optional
            Torch device to use. If None, falls back to `self.device`.
        """
        device = torch.device(self.device if device is None else device)
        # Sum of scheduled epochs
        max_epochs = sum(self.run_config[my_config_dict.run_config_model][my_config_dict.model_epochs_sequence])
        log_steps = self.run_config[my_config_dict.run_config_general][my_config_dict.general_logging_steps]
        
        # Settings for early stopping
        early_stop = self.run_config[my_config_dict.run_config_general][my_config_dict.general_early_stop]
        stop1 = early_stop[my_config_dict.general_tolerance_mrad]
        stop2 = early_stop[my_config_dict.general_least_improvement_mrad]
        
        splits = self.datasplitter.splits
        for split_type in splits:
            for split_size, split_df in splits[split_type].items():
                log.info(f"Begin calibration for dataset split {split_type}: size {split_size}")
                
                run = f"{split_type}_{split_size}"
                self.tb_logger = TensorboardLogger(run=run, heliostat_names=self.heliostat_ids, log_dir=self.save_directory / 'logs')
                
                # Set gradients to False for selection of frozen parameters
                freeze_and_phase = self.run_config[my_config_dict.run_config_model][my_config_dict.model_apply_with_caution]
                freeze_params = freeze_and_phase[my_config_dict.model_freeze_params]
                if len(freeze_params) > 0:
                    self.freeze_params(freeze_params)  # freeze params for this run
                    
                # Check, whether to perform phased learning / training
                phased_learning_bool = bool(freeze_and_phase[my_config_dict.model_phase_param_learning])
                phase_counter = 0  # required to track the phases for learning

                # Empty validation error history
                self.val_error_history = []
                best_val_metric = float('inf')
                best_state_dict = None
                
                # Log initial parameter state
                self.tb_logger.log_parameters_obj(-1, self.learnable_params_dict)
                
                # Set early stopping if worst error drops below tolerance
                for epoch in range(max_epochs):
                    # Apply phased learning if configured
                    if phased_learning_bool:
                        phase_counter = self.phase_freeze_training(current_epoch=epoch, phase_counter=phase_counter)
                    
                    # Train
                    self.train()
                    split_ids_train = self.datasplitter.get_helio_and_calib_ids_from_split(split_type, split_size, 'train', heliostat_ids=self.heliostat_ids)
                    self.tb_logger.set_mode("Train")
                    self.tb_logger.set_helio_and_calib_ids(split_ids_train)
                    data_batch_train = self.dataloader.get_field_batch(helio_and_calib_ids=split_ids_train, device=device)
                    field_orientations, pred_flux_bitmaps = self.forward(data_batch_train)
                    train_loss, train_alignment_errors, _ = self.evaluate_model(epoch, field_orientations, pred_flux_bitmaps, data_batch_train)             
                
                    # Backward
                    self.optimizer.zero_grad()
                    train_loss.backward()
                    nan_grad_detected = check_for_nan_grad(self.learnable_params_dict)
                    self.optimizer.step()
                    
                    # Log parameters
                    self.tb_logger.log_parameters_obj(epoch, self.learnable_params_dict)
                        
                    # Validation
                    self.eval()
                    split_ids_val = self.datasplitter.get_helio_and_calib_ids_from_split(split_type, split_size, 'validation', heliostat_ids=self.heliostat_ids)
                    self.tb_logger.set_mode("Validation")
                    self.tb_logger.set_helio_and_calib_ids(split_ids_val)
                    with torch.no_grad():
                        data_batch_val = self.dataloader.get_field_batch(helio_and_calib_ids=split_ids_val, device=device)
                        val_orientations, val_pred_bitmaps = self.forward(data_batch_val)
                        val_loss, val_alignment_errors, _ = self.evaluate_model(epoch, val_orientations, val_pred_bitmaps, data_batch_val)
                        best_val_metric, best_state_dict = self.update_best_parameters(best_val_metric, val_alignment_errors.mean(), best_state_dict)

                    # Step schedulers if not None
                    if self.schedulers:
                        if isinstance(self.schedulers, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.schedulers.step(val_loss)
                            current_lr = self.optimizer.param_groups[0]['lr']
                        else:
                            self.schedulers.step()
                            current_lr = self.schedulers.get_last_lr()[0]
                    else:
                        current_lr = self.optimizer.param_groups[0]['lr']
                    self.tb_logger.log_metric(epoch, 'BaseLearningRate', current_lr)
                    
                    if epoch % log_steps == 0:
                        log.info(f"Epoch: {epoch} of {max_epochs}, "
                         f"Train / Val Loss: {train_loss.item():.8f} / {val_loss.item():.8f}, "
                         f"Train / Val Avg.Error (mrad): {train_alignment_errors.mean().item():.2f} / {val_alignment_errors.mean().item():.2f}, "
                         f"LR: {current_lr: 8f}")
                    
                    if (self.check_early_stopping(val_alignment_errors, stop_1=stop1, stop_2=stop2) or torch.isnan(val_orientations).any() or nan_grad_detected):
                        break  # break training loop if early stopping is triggered or parameters contain nan-gradients
                    
                    # Save backup csv for every 100 epochs
                    if epoch % 100 == 0:
                        self.tb_logger.save_dataframes_to_csv()
                        
                    # Delete some objects to save RAM (seems necessary, otherwise CUDA out of memory error)
                    torch.cuda.empty_cache()
                    del train_loss
                    del val_loss
                    del pred_flux_bitmaps
                    del val_pred_bitmaps
                    del field_orientations
                    del val_orientations
                    
                # Testing
                self.load_best_parameters(best_state_dict)
                self.eval()
                split_ids_test = self.datasplitter.get_helio_and_calib_ids_from_split(split_type, split_size, 'test', heliostat_ids=self.heliostat_ids)
                self.tb_logger.set_mode("Test")
                self.tb_logger.set_helio_and_calib_ids(split_ids_test)
                data_batch_test = self.dataloader.get_field_batch(helio_and_calib_ids=split_ids_test, device=device)
                test_orientations, test_pred_bitmaps = self.forward(data_batch_test)
                _, test_alignment_errors, actual_alignment_errors = self.evaluate_model(epoch, test_orientations, test_pred_bitmaps, data_batch_test)
                log.info(f"Finally: Test Avg.Error (mrad): {test_alignment_errors.mean().item():.2f}, Actual Avg.Error (mrad): {actual_alignment_errors.mean().item():.2f}")
                
                self.tb_logger.close()
     
    def forward(self, field_batch: List[Dict[str, Union[torch.Tensor, str]]], device: Optional[Union[torch.device, str]] = None) -> Tuple[torch.Tensor]:
        """
        Forward pass to produce per‑sample heliostat orientations and (optionally)
        ray‑traced flux bitmaps.

        Parameters
        ----------
        field_batch : list of dict
            Batch of field samples. Each dict contains (per sample):
            - motor positions  (key: my_config_dict.field_sample_motor_positions)
            - incident ray directions (key: my_config_dict.field_sample_incident_rays)
            - target area names (key: my_config_dict.field_sample_target_names)
        device : torch.device or str, optional
            Torch device to use. If None, falls back to `self.device`.

        Returns
        -------
        field_orientations : torch.Tensor
            Shape [B, H, 4, 4]. Homogeneous transforms of each heliostat per batch
            sample (B = batch size, H = number of heliostats).
        pred_flux_bitmaps : torch.Tensor
            Shape [B, H, He, Hu]. Predicted flux images if `self.use_raytracing` is
            True; zeros otherwise. Resolution equals raytracer bitmap resolution.
        """
        device = torch.device(self.device if device is None else device)
        
        batch_size = len(field_batch)  # number of samples per heliostat in field  
        heliostat_field = self.scenario.heliostat_field
        num_helio = heliostat_field.number_of_heliostats
        
        raytracer = self.raytracer
        
        # Initialize empty outputs
        field_orientations = torch.zeros((batch_size, num_helio, 4, 4), device=device)
        pred_flux_bitmaps = torch.zeros((batch_size, num_helio, raytracer.bitmap_resolution_e, raytracer.bitmap_resolution_u), device=device)
        
        # Iterate over samples of Heliostat field data
        for sample, data in enumerate(field_batch):
            # Align Heliostat field with data on motor positions
            heliostat_field.align_surfaces_with_motor_positions(motor_positions=data[my_config_dict.field_sample_motor_positions],
                                                                device=device)
            # Save Tensor for the orientations of Heliostats
            field_orientations[sample] = heliostat_field.rigid_body_kinematic.orientations
            if torch.isnan(field_orientations[sample]).any():
                print(f"Orientations are NaN in sampe {sample}")
            
            # Perform Raytracing if configured
            if self.use_raytracing:
                incident_ray_directions = data[my_config_dict.field_sample_incident_rays]
                target_area_names = data[my_config_dict.field_sample_target_names]
                target_areas = [self.target_areas[name] for name in target_area_names]
                sample_flux_bitmaps = raytracer.trace_rays_separate(incident_ray_directions=incident_ray_directions,
                                                                    target_areas=target_areas,
                                                                    device=device)
                pred_flux_bitmaps[sample] = sample_flux_bitmaps
        return field_orientations, pred_flux_bitmaps
    
    def evaluate_model(self, epoch: int, orientations: torch.Tensor, pred_flux_bitmaps: torch.Tensor, field_batch: List[Dict]) -> Tuple[torch.Tensor]:
        """
        Compute the training objective (scalar loss) and evaluation metrics for a batch.

        Workflow
        --------
        1) Build ground‑truth reflection directions from measured (or COM‑derived)
           flux centers. If simulated data provides ideal centers, optionally use
           them as loss ground truth (config‑dependent), but **never** for the
           evaluation metric.
        2) Build predicted reflection directions either
           (a) from model orientations and incident rays, or
           (b) from predicted flux centers (COM) if `loss_basis` = center_of_mass.
           Empty predicted flux bitmaps automatically fall back to (a).
        3) Compute alignment‑loss per heliostat from cosine similarities.
        4) If contour loss is enabled (and past warmup), compute fine/coarse/center
           image losses, blend them with a soft transition, and apply a guardrail
           that reverts to alignment loss for diverging heliostats.

        Parameters
        ----------
        epoch : int
            Global training epoch index.
        orientations : torch.Tensor
            Shape [B, H, 4, 4]. Homogeneous transforms of each heliostat.
        pred_flux_bitmaps : torch.Tensor
            Shape [B, H, He, Hu]. Predicted flux images (normalized upstream).
        field_batch : list of dict
            Batch entries holding incident rays, target names, flux images and/or
            flux centers; simulated data may additionally hold ideal centers.

        Returns
        -------
        loss : torch.Tensor
            Scalar mean loss over heliostats (averaged from per‑heliostat guardrail
            loss).
        eval_alignment_errors : torch.Tensor
            Shape [B, H] in mrad. Alignment errors between **predicted vs. measured**
            reflection directions (evaluation metric).
        actual_alignment_errors : torch.Tensor
            Shape [B, H] in mrad. Only for simulated data: alignment errors between
            **predicted vs. ideal** reflection directions; zeros otherwise.
        """
        # TODO: Normalize true and predicted Bitmaps!
        # Get nested list of target area names and stacked Tensor of incident ray directions from batch data
        target_areas = [sample[my_config_dict.field_sample_target_names] for sample in field_batch]
        incident_ray_directions = torch.stack([sample[my_config_dict.field_sample_incident_rays] for sample in field_batch])
        
        # Load config for loss
        loss_config = self.run_config[my_config_dict.run_config_general][my_config_dict.general_loss_config]
        
        # All inputs must have the same batch size
        assert len(target_areas) == orientations.shape[0] == pred_flux_bitmaps.shape[0],\
            "Batch sizes must be equal for target_areas, orientations and pred_flux_bitmaps."
        
        # 1) Calculate model metric: Alignment Errors from True vs Pred Reflection Directions
        # 1a) Ground-Truth centroid
        with torch.no_grad(): 
            if self.dataloader.is_simulated_data:  # training data was simulated and holds data for ideal centroid
                true_ideal_flux_centers = torch.stack([sample[my_config_dict.field_sample_ideal_flux_centers] 
                                                       for sample in field_batch])
                true_ideal_reflection_directions = self.calc_reflection_directions_from_flux_centers(
                        true_ideal_flux_centers, orientations)
            
            if self.dataloader.load_flux_images:  # re-calculate flux-centers for true flux centroids as COM
                true_flux_bitmaps = torch.stack([sample[my_config_dict.field_sample_flux_images] 
                                                 for sample in field_batch])
                true_flux_centers = self.calc_centers_of_mass(bitmaps=true_flux_bitmaps, target_area_names=target_areas, threshold=0.0)
            else:  # use centroid coordinates from calibration config-files
                true_flux_centers = torch.stack([sample[my_config_dict.field_sample_flux_centers] 
                                                 for sample in field_batch])
    
        # 1b) Reflection Directions for Ground-Truth and Prediction
        true_reflection_directions = self.calc_reflection_directions_from_flux_centers(true_flux_centers, orientations)
        pred_reflection_directions = self.calc_reflection_directions_from_orientations(incident_ray_directions, orientations)
        with torch.no_grad(): # for evaluation metric never use 'ideal' centroid as GT
            eval_alignment_errors = self.calc_alignment_errors_stable(pred_reflection_directions, true_reflection_directions)
            self.tb_logger.log_alignment_errors(epoch=epoch, alignment_errors=eval_alignment_errors)
        
        # 2) Calculate Alignment Loss based on configurations (Can be based on vectorial alignments or based on flux images)
        loss_config = self.run_config[my_config_dict.run_config_general][my_config_dict.general_loss_config]
        alignment_loss_basis = loss_config[my_config_dict.general_loss_basis].lower()
        
        # 2a) re-calculate GT of reflection axis if set to "ideal"
        if (loss_config[my_config_dict.general_loss_groundtruth] == "ideal"  # use ideal centroids in loss
                and self.dataloader.is_simulated_data):
                true_reflection_directions = true_ideal_reflection_directions
                
        if "orientation" in alignment_loss_basis:
            cosine_similarities = self.calc_cosine_similarities(pred_reflection_directions, true_reflection_directions)
 
        elif ("center_of_mass" in alignment_loss_basis) or ("com" in alignment_loss_basis):
            empty_mask, empty_count = self.count_empty_flux_predictions(pred_flux_bitmaps, epoch=epoch, threshold=1)
            mass_centers = self.calc_centers_of_mass(bitmaps=pred_flux_bitmaps, target_area_names=target_areas, threshold=0.0)
            pred_reflection_directions = self.calc_reflection_directions_from_flux_centers(mass_centers, orientations)
            if empty_count > 0:  # use fallback meachnism if any flux predictions are empty
                fallback_reflections = self.calc_reflection_directions_from_orientations(
                    incident_ray_directions=incident_ray_directions, orientations=orientations)
                pred_reflection_directions = torch.where(empty_mask.unsqueeze(-1),  fallback_reflections, pred_reflection_directions)
            cosine_similarities = self.calc_cosine_similarities(pred_reflection_directions, true_reflection_directions)
                   
        else:
            raise ValueError(f"Unsupported value for alignment_loss_basis: {alignment_loss_basis}")   
        
        alignment_loss = self.calc_alignment_loss(cosine_similarities) * 1e6  # loss per helisotat [H_idx]
        self.tb_logger.log_loss(f"AlignmentLoss_{loss_config[my_config_dict.general_loss_type]}", alignment_loss.mean().item(), epoch)
        
        # 3) Calculate the "actual" alignment errors if ideal flux center is known, ie. data was simulated
        actual_alignment_errors = torch.zeros_like(eval_alignment_errors)
        if self.dataloader.is_simulated_data:  # data should contain the ideal flux centers
            with torch.no_grad():
                # Calculate the "actual" alignment errors based on the known reflection axis
                actual_alignment_errors = self.calc_alignment_errors_stable(pred_reflection_directions, true_ideal_reflection_directions)
            self.tb_logger.log_alignment_errors(epoch=epoch, alignment_errors=actual_alignment_errors, is_actual=True)
        
        # 4) Calculate Image-based Loss if this is configured 
        guardrail_loss = alignment_loss
        if self.use_raytracing:
            # Check if contour-based loss is configured
            contour_config = loss_config[my_config_dict.general_contour_config]
            if bool(contour_config[my_config_dict.general_use_contour]): 
                if  epoch > contour_config[my_config_dict.general_contour_warmup]:
                    # One loss term for fine alignment (requires overlap) and one for coarse alignment (no overlap of contours) 
                    fine_loss, coarse_loss, center_loss = self.calc_contour_loss(pred_flux_bitmaps, true_flux_bitmaps, target_areas, epoch)
                    combined_loss = self.soft_loss_transition(alignment_loss, fine_loss, coarse_loss, center_loss, epoch) # [H_idx]
                    guardrail_loss = self.guardrail_loss(eval_alignment_errors, combined_loss, alignment_loss) # [H_idx]
                    self.tb_logger.log_loss(f"CombinedLoss", combined_loss.mean().item(), epoch)
                self.tb_logger.log_loss(f"GuardrailLoss", guardrail_loss.mean().item(), epoch)
                
        return guardrail_loss.mean(), eval_alignment_errors, actual_alignment_errors

    def check_early_stopping(self, val_alignment_errors: torch.Tensor, stop_1: float=0.5, stop_2: float=0.05) -> bool:
        """
        Evaluate early‑stopping conditions from validation alignment errors.

        Parameters
        ----------
        val_alignment_errors : torch.Tensor
            Shape [B, H] (mrad). Validation batch alignment errors.
        stop_1 : float
            Threshold on worst mean heliostat error to stop early.
        stop_2 : float
            Minimum required improvement of mean validation error over the recent
            window (fixed length) to continue; otherwise stop.

        Returns
        -------
        bool
            True if an early‑stopping criterion is met; False otherwise.
        """
        # Compute mean alignment error per heliostat [H]
        mean_error_per_heliostat = val_alignment_errors.mean(dim=0)  # shape: [H]
        worst_heliostat_error = mean_error_per_heliostat.max().item()
        mean_val_error = mean_error_per_heliostat.mean().item()
        self.val_error_history.append(mean_val_error)

        # Criterion 1: Worst individual heliostat below threshold
        if worst_heliostat_error < stop_1:
            log.info(f"Early stopping (1): Worst heliostat error ({worst_heliostat_error:.3f} mrad) is below threshold.")
            return True

        # Criterion 2: No sufficient improvement in last 100 epochs
        if len(self.val_error_history) >= 500:
            improvement = self.val_error_history[-500] - self.val_error_history[-1]
            if improvement < stop_2 and stop_2 > 0:
                log.info(f"Early stopping (2): Mean validation error did not improve by {stop_2} mrad over last 500 epochs (Δ={improvement:.4f}).")
                return True
        return False
        
    def soft_loss_transition(self, alignment_loss: torch.Tensor, fine_contour_loss: torch.Tensor, coarse_contour_loss: torch.Tensor, center_loss: torch.Tensor,
                             epoch: int, beta: float=0.6, omega: float=0.05, gamma: float=0.2) -> torch.Tensor:
        """
        Blend alignment and contour‑based losses with a two‑phase schedule.

        During warmup → midpoint, linearly increase the contour contribution.
        After midpoint, use a fixed mixture of fine/coarse contour losses plus a
        small alignment regularizer. Optionally add a center‑of‑mass term.

        Parameters
        ----------
        alignment_loss : torch.Tensor
            Shape [H]. Per‑heliostat alignment loss.
        fine_contour_loss : torch.Tensor
            Shape [H]. Loss that assumes overlapping contours (fine alignment).
        coarse_contour_loss : torch.Tensor
            Shape [H]. Loss robust to non‑overlapping contours (coarser stage).
        center_loss : torch.Tensor
            Shape [H]. Optional COM‑distance term (per heliostat).
        epoch : int
            Global epoch, used to compute the blending coefficient.
        beta : float
            Weight between fine/coarse contour losses (coarse weight = beta).
        omega : float
            Weight for center‑of‑mass loss (if enabled).
        gamma : float
            Residual weight for the alignment loss.

        Returns
        -------
        torch.Tensor
            Shape [H]. Blended loss per heliostat.
        """
        assert alignment_loss.shape == fine_contour_loss.shape, \
            f"Loss terms need to have equal shapes, but have {alignment_loss.shape}, {fine_contour_loss.shape}"
        assert alignment_loss.shape == coarse_contour_loss.shape, \
            f"Loss terms need to have equal shapes, but have {alignment_loss.shape}, {fine_contour_loss.shape}" 
                
        max_epochs = sum(self.run_config[my_config_dict.run_config_model][my_config_dict.model_epochs_sequence])
        contour_config = self.run_config[my_config_dict.run_config_general][my_config_dict.general_loss_config][my_config_dict.general_contour_config]
        use_center_loss = bool(contour_config[my_config_dict.general_use_center])
        warmup = contour_config[my_config_dict.general_contour_warmup]
        half = (max_epochs - warmup) / 2 + warmup
        
        # Make soft transition to contour loss terms in first half
        if epoch <= half:
            # alpha goes from 0 at 'warmup' to 1 at 'half' 
            alpha = (epoch - warmup) / (half - warmup) 
            combined_loss = alpha * ((1-beta) * fine_contour_loss + beta * coarse_contour_loss) + (1 - alpha + gamma) * alignment_loss
        else: 
            combined_loss = (1-beta) * fine_contour_loss + beta * coarse_contour_loss +  gamma * alignment_loss
            
        if use_center_loss:
            combined_loss += omega * center_loss
            self.tb_logger.log_loss(f"ContourLossCenters", center_loss.mean().item(), epoch)  
                    
        # Log Losses
        fine_loss_key = contour_config[my_config_dict.general_contour_fine].upper()
        if fine_loss_key != "0":
            self.tb_logger.log_loss(f"ContourLoss{fine_loss_key}", fine_contour_loss.mean().item(), epoch)
        coarse_loss_key = contour_config[my_config_dict.general_contour_coarse].upper()
        if coarse_loss_key != "0":
            self.tb_logger.log_loss(f"ContourLoss{coarse_loss_key}", coarse_contour_loss.mean().item(), epoch)                    
        return combined_loss
    
    def guardrail_loss(self, alignment_errors: torch.Tensor, combined_loss: torch.Tensor, alignment_loss: torch.Tensor, threshold=8.0) -> torch.Tensor:
        """
        Guardrail: revert to alignment loss for heliostats with large mean errors.

        Parameters
        ----------
        alignment_errors : torch.Tensor
            Shape [B, H] (mrad). Per‑sample per‑heliostat evaluation errors.
        combined_loss : torch.Tensor
            Shape [H]. Current mixed loss per heliostat.
        alignment_loss : torch.Tensor
            Shape [H]. Alignment loss per heliostat to fall back to.
        threshold : float
            Mean error threshold (mrad) above which the guardrail activates.

        Returns
        -------
        torch.Tensor
            Shape [H]. Final per‑heliostat loss after the guardrail switch.
        """
        H = len(self.heliostat_ids)
        assert combined_loss.shape == alignment_loss.shape, \
            f"Loss terms need to have equal shapes, but have {combined_loss.shape}, {alignment_loss.shape}"
        assert H == combined_loss.shape[0] == alignment_loss.shape[0] == alignment_errors.shape[1], \
            f"Loss terms need be per Heliostat and given as shape [H] = {H}"

        # Take mean for each Heliostat
        alignment_errors = alignment_errors.mean(dim=0)  # shape [H]

        # Determine fallback mask based on alignment errors
        fallback_mask = (alignment_errors > threshold)

        # Apply fallback where alignment error too high
        guardrail_loss = torch.where(fallback_mask, alignment_loss, combined_loss)
        return guardrail_loss
        
    def calc_alignment_loss(self, cosine_similarities: torch.Tensor) -> torch.Tensor:
        """
        Map cosine similarities to a per‑heliostat loss according to config.

        Supported loss types (case‑insensitive):
        - 'MAE'/'L1', 'MSE'/'L2', 'MAX' (L∞), 'HUBER[-_<δ>]', 'QUANTILE'/'PERCENTILE'[-_<q>]

        Parameters
        ----------
        cosine_similarities : torch.Tensor
            Shape [B, H], values in [-1, 1].

        Returns
        -------
        torch.Tensor
            Shape [H]. Per‑heliostat loss averaged over batch (or the chosen
            statistic for MAX/QUANTILE).
        """
        # Add scaling factor for loss
        sf_loss = 1e6
        
        # Get the loss type configurations 
        loss_config = self.run_config[my_config_dict.run_config_general][my_config_dict.general_loss_config]
        loss_type = loss_config[my_config_dict.general_loss_type].upper()
        
        # Perfect alignment = 1.0
        target = torch.ones_like(cosine_similarities)  

        # L1 / MAE Loss per Heliostat
        if any(key in loss_type for key in ['MAE', 'L1']):
            # Compute mean absolute error per heliostat across batch dimension
            loss_per_heliostat = torch.mean(torch.abs(cosine_similarities - target), dim=0)  # [H_idx]

        # L2 / MSE Loss per heliostat
        elif any(key in loss_type for key in ['MSE', 'L2']):
            loss_per_heliostat = torch.mean((cosine_similarities - target) ** 2, dim=0)  # [H_idx]

        # MAX error (L∞ loss) per heliostat
        elif 'MAX' in loss_type:
            loss_per_heliostat = torch.max(torch.abs(cosine_similarities - target), dim=0).values  # [H_idx]
        
        # Huber implementation with tunable delta
        elif 'HUBER' in loss_type:
            # Extract delta from string using regex, fallback to default if not found
            match = re.search(r'HUBER[-_]?(\d*\.?\d+)', loss_type)
            delta = float(match.group(1)) if match else 0.00005  # default ≈ 10 mrad

            diff = cosine_similarities - target
            abs_diff = torch.abs(diff)
            huber = torch.where(abs_diff < delta, 0.5 * diff**2, delta * (abs_diff - 0.5 * delta))
            loss_per_heliostat = huber.mean(dim=0)

        # Quantile or Percentile implementation with tunable value
        elif 'QUANTILE' in loss_type or 'PERCENTILE' in loss_type:
            match = re.search(r'(QUANTILE|PERCENTILE)[-_]?(\d+)', loss_type)
            quantile_val = float(match.group(2)) if match else 90.0  # default 90
            quantile = quantile_val / 100.0
            loss_per_heliostat = torch.quantile(torch.abs(cosine_similarities - target), q=quantile, dim=0)

        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        return loss_per_heliostat
    
    def calc_contour_loss(self, pred_bitmaps: torch.Tensor, true_bitmaps: torch.Tensor, target_areas, epoch, 
                          threshold=0.6, sharpness=70.0, num_interpolate=4, sigma_in=15.0, sigma_out=25.0) -> Tuple[torch.Tensor]:
        """
        Compute image‑based losses from upper contours of predicted and true flux images.

        For each heliostat:
        1) Pre‑smooth, extract upper contours with a soft detector, post‑smooth.
        2) Compute a fine‑alignment loss (requiring overlap) and a coarse loss
           (robust when contours do not overlap).
        3) Optionally compute a COM distance between predicted vs. true contours.

        Parameters
        ----------
        pred_bitmaps, true_bitmaps : torch.Tensor
            Shape [B, H, He, Hu]. Predicted and ground‑truth flux images.
        target_areas : list[list[str]]
            Nested list of target names per sample used for COM computation.
        epoch : int
            Global epoch for logging.
        threshold : float
            Soft threshold for contour detection.
        sharpness : float
            Steepness parameter for soft contour extraction.
        num_interpolate : int
            Pre/post up‑down sampling count for smoothing continuity.
        sigma_in, sigma_out : float
            Gaussian blur std dev before and after contour detection.

        Returns
        -------
        fine_loss : torch.Tensor
            Shape [H]. Fine contour loss per heliostat.
        coarse_loss : torch.Tensor
            Shape [H]. Coarse contour loss per heliostat.
        center_loss : torch.Tensor
            Shape [H]. Optional COM loss per heliostat (zeros if disabled).
        """
        assert pred_bitmaps.shape == true_bitmaps.shape, \
            f"pred_bitmaps and true_bitmaps must have equal shapes, but have {pred_bitmaps.shape} and {true_bitmaps.shape}"
        
        B, H = pred_bitmaps.shape[0], pred_bitmaps.shape[1]
        device = pred_bitmaps.device
        fine_loss_per_heliostat = torch.zeros(H, device=device)
        coarse_loss_per_heliostat = torch.zeros(H, device=device)
        center_loss_per_heliostat = torch.zeros(H, device=device)
        
        # Get the contour loss configurations
        contour_config = self.run_config[my_config_dict.run_config_general][my_config_dict.general_loss_config][my_config_dict.general_contour_config]
        warmup = contour_config[my_config_dict.general_contour_warmup]
        use_centers = bool(contour_config[my_config_dict.general_use_center])
        
        # Scale to similar magnitudes      
        loss_scaling = {"MSE": 1, "CHD": 1e-3, "SDF": 1e-1, "DICE": 1e-2}
        
        # Configure fine loss function
        fine_loss_key = contour_config[my_config_dict.general_contour_fine].upper()
        if fine_loss_key == "0":
            fine_loss_fnct = lambda x, y: 0 
        elif fine_loss_key == "MSE":
            mse_fnc = torch.nn.MSELoss()
            fine_loss_fnct = lambda x, y: mse_fnc(x, y) * loss_scaling[fine_loss_key]
        elif fine_loss_key == "DICE":
            fine_loss_fnct = lambda x, y: dice_loss(x, y) * loss_scaling[fine_loss_key]
        elif "MSE" in fine_loss_key and "DICE" in fine_loss_key:
            mse_fnc = torch.nn.MSELoss()
            fine_loss_fnct = lambda x, y: (mse_fnc(x, y) * loss_scaling["MSE"] 
                                           + dice_loss(x, y) * loss_scaling["DICE"])
        else: 
            raise ValueError(f"Unsupported type for fine contour loss: {fine_loss_key}")    
        
        # Configure coarse loss function
        coarse_loss_key = contour_config[my_config_dict.general_contour_coarse].upper()
        if coarse_loss_key == "0":
            coarse_loss_fnct = lambda x, y: 0 
        elif coarse_loss_key == "CHD":
            coarse_loss_fnct = lambda x, y: chamfer_distance_batch_optimized(x, y) * loss_scaling[coarse_loss_key]
        elif coarse_loss_key == "SDF":
            coarse_loss_fnct = lambda x, y: sdf_loss(x, y) * loss_scaling[coarse_loss_key]
        elif ("CHD" in coarse_loss_key and "SDF" in coarse_loss_key):
            coarse_loss_fnct = lambda x, y: (chamfer_distance_batch_optimized(x, y) * loss_scaling["CHD"] 
                                           + sdf_loss(x, y) * loss_scaling["SDF"])
        else: 
            raise ValueError(f"Unsupported type for coarse contour loss: {coarse_loss_key}")    
        
        # Iterate over Heliostat indices
        all_pred_contours = torch.zeros_like(pred_bitmaps)
        all_true_contours = torch.zeros_like(true_bitmaps)
        for h_idx in range(H):
            pred_contours = self.find_upper_contour(pred_bitmaps[:, h_idx], threshold, sharpness, num_interpolate, sigma_in, sigma_out)
            with torch.no_grad():
                true_contours = self.find_upper_contour(true_bitmaps[:, h_idx], threshold, sharpness, num_interpolate, sigma_in, sigma_out)
            fine_loss_per_heliostat[h_idx] = fine_loss_fnct(pred_contours, true_contours).mean()
            coarse_loss_per_heliostat[h_idx] = coarse_loss_fnct(pred_contours, true_contours).mean()
            if use_centers:
                hel_target_areas = [sample[h_idx] for sample in target_areas]
                center_loss_per_heliostat[h_idx] = self.calc_center_mass_loss(pred_contours, true_contours, hel_target_areas)
            with torch.no_grad():
                all_pred_contours[:, h_idx] = pred_contours
                all_true_contours[:, h_idx] = true_contours
            
        # Log contour images
        if epoch % 50 == 0 or epoch == (warmup + 1): # Log on initiation and every 100th epoch         
            self.tb_logger.log_flux_bitmaps(epoch, all_pred_contours - all_true_contours, type='DiffContour1', threshold=0)
        #    self.tb_logger.log_diff_flux_bitmaps(epoch, all_pred_contours, all_true_contours, type='DiffContour', normalize=True, threshold=0.0001)     
        del all_pred_contours
        del all_true_contours
        return fine_loss_per_heliostat, coarse_loss_per_heliostat, center_loss_per_heliostat
    
    def calc_center_mass_loss(self, images1: torch.Tensor, images2: torch.Tensor, target_areas_names: List[str], threshold=0.1) -> torch.Tensor:
        """
        Distance between centers of mass (COM) of two contour (or flux) images.

        Parameters
        ----------
        images1, images2 : torch.Tensor
            Shape [B, He, Hu], normalized to [0, 1]. Must have identical shape.
        target_areas_names : list[str]
            Target IDs for each sample to resolve plane bases and centers.
        threshold : float
            Minimum intensity used in COM computation.

        Returns
        -------
        torch.Tensor
            Scalar mean L2 distance between COMs over the batch.
        """
        assert images1.shape == images2.shape
        device = images1.device
        center_of_mass1 = []
        center_of_mass2 = []
        
        for i in range(images1.shape[0]):
            target_area =  self.target_areas[target_areas_names[i]]
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
           
    def calc_flux_centers_from_orientations(self, orientations: torch.Tensor, incident_ray_directions: torch.Tensor, target_area_names: List[List[str]]) -> torch.Tensor:
        """
        Intersect reflected rays with target planes to obtain flux center points.

        Parameters
        ----------
        orientations : torch.Tensor
            Shape [B, H, 4, 4]. Heliostat homogeneous transforms.
        incident_ray_directions : torch.Tensor
            Shape [B, H, 4]. Incoming ray directions (homogeneous).
        target_area_names : list[list[str]]
            Target identifiers per sample/heliostat to select plane definitions.

        Returns
        -------
        torch.Tensor
            Shape [B, H, 4]. Flux center points on the target planes (homogeneous).
        """
        # Initiate output Tensor
        device = orientations.device
        incident_ray_directions = incident_ray_directions.to(device)
        flux_centers = torch.zeros(orientations.shape[:3], dtype=orientations.dtype, device=device, requires_grad=True)
        
        # Add dimensions batchsize and Heliostat index if not given
        while (d := orientations.dim()) < 4: orientations = orientations.unsqueeze(0) # to [B, H, 4, 4] 
        while (d := incident_ray_directions.dim()) < 3: incident_ray_directions = incident_ray_directions.unsqueeze(0) # to [B, H, 4] 
        while not isinstance(target_area_names[0], list()):
            target_area_names = list(target_area_names) # should be a list of lists

        # Iterate over field samples
        for s, target_areas_sample in enumerate(target_area_names):
            for h, target_area_name in enumerate(target_areas_sample):
                concentrator_normal = orientations[s, h, 0:4, 2]
                concentrator_origin = orientations[s, h, 0:4, 3]
                reflected_ray = raytracing_utils.reflect(incoming_ray_direction=incident_ray_directions[s, h],
                                                         reflection_surface_normals=concentrator_normal)
                
                target_area = self.target_areas[target_area_name]
                center_point, t = calculate_intersection(ray_origin=concentrator_origin,
                                                         ray_direction=reflected_ray,
                                                         plane_center=target_area.center.to(device),
                                                         plane_normal=target_area.normal_vector.to(device))
                flux_centers[s, h] = center_point             
        return flux_centers
    
    @staticmethod
    def calc_reflection_directions_from_orientations(incident_ray_directions: torch.Tensor, orientations: torch.Tensor) -> torch.Tensor:
        """
        Compute reflected ray directions from incident rays and heliostat normals.

        Parameters
        ----------
        incident_ray_directions : torch.Tensor
            Shape [B, H, 4] or broadcastable. Incoming directions (homogeneous).
        orientations : torch.Tensor
            Shape [B, H, 4, 4] or broadcastable. Homogeneous transforms; the z‑axis
            (column 2) encodes the mirror normal in world coordinates.

        Returns
        -------
        torch.Tensor
            Shape [B, H, 4]. Reflected ray directions (homogeneous).
        """
        # Add dimensions if necessary
        while incident_ray_directions.dim() < 3:
            incident_ray_directions = incident_ray_directions.unsqueeze(0)  # [B, H, 4]
        while orientations.dim() < 4:
            orientations = orientations.unsqueeze(0)  # [B, H, 4, 4]

        # Extract surface normal vectors from orientation matrices (3rd column)
        normals = orientations[:, :, :, 2]  # [B, H, 4]
        
        # Reflect incoming rays over normals using the reflection formula
        # r = d - 2 * (d · n) * n
        dot = torch.sum(incident_ray_directions * normals, dim=-1, keepdim=True)  # [B, H, 1]
        reflected = incident_ray_directions - 2 * dot * normals  # [B, H, 4]
        return reflected
    
    def calc_centers_of_mass(self, bitmaps: torch.Tensor, target_area_names: List[List[str]], threshold=0) -> torch.Tensor:
        """
        Compute the flux focal spot centers of mass (CoM) in ENU coordinates for a batch of flux images.

        Each bitmap is processed relative to its assigned target area geometry, so that the pixel-based
        center of mass is mapped into absolute 4D ENU coordinates. This enables consistent comparison of
        measured and predicted flux centers across different heliostat–target configurations.

        Parameters
        ----------
        bitmaps : torch.Tensor
            Flux intensity images with shape [B, H, H_img, W_img], where B is the batch size and H the
            number of heliostats. If fewer dimensions are provided, batch and/or heliostat dimensions
            are automatically added.
        target_area_names : List[List[str]]
            Nested list of target area names per sample and heliostat (shape [B, H]).
            Each name must correspond to an entry in `self.target_areas`.
        threshold : float, optional
            Intensity threshold applied when computing the center of mass. Pixels below this value
            are ignored to suppress noise and background (default=0).

        Returns
        -------
        centers_of_mass : torch.Tensor
            Tensor of shape [B, H, 4], where each entry is the ENU 4D coordinate of the flux focal
            spot center of mass for the respective heliostat and batch sample.
        """
        device = bitmaps.device
        
        # Initiate output Tensor
        centers_of_mass = torch.zeros((bitmaps.shape[0], bitmaps.shape[1], 4), dtype=bitmaps.dtype, device=device)
        
        # Add dimensions batchsize and Heliostat index if not given
        while (d := bitmaps.dim()) < 4: bitmaps = bitmaps.unsqueeze(0) # to [B, H, (res)] 
        while not isinstance(target_area_names[0], List):
            target_area_names = list(target_area_names) # should be a list of lists
        
        # Iterate over field samples
        for s, target_areas_sample in enumerate(target_area_names):
            for h, target_area_name in enumerate(target_areas_sample):
                target_area = self.target_areas[target_area_name]
                centers_of_mass[s, h, 0:4] = get_center_of_mass(bitmap=bitmaps[s, h], 
                                                                target_center=target_area.center,
                                                                plane_e=target_area.plane_e,
                                                                plane_u=target_area.plane_u,
                                                                threshold=threshold,
                                                                device=device)
        return centers_of_mass
    
    @staticmethod
    def calc_reflection_directions_from_flux_centers(flux_centers: torch.Tensor, orientations: torch.Tensor):
        """
        
        """
        # Add dimensions batchsize and Heliostat index if not given
        while (d := flux_centers.dim()) < 3: flux_centers = flux_centers.unsqueeze(0) # to [B, H, 4] 
        while (d := orientations.dim()) < 4: orientations = orientations.unsqueeze(0) # to [B, H, 4, 4] 

        # Origin coordinates can be extracted from orientations
        concentrator_origins = orientations[:, :, 0:4, 3]
        assert flux_centers.shape == concentrator_origins.shape, \
            "Given flux_centers and concentrator_origins should have equal shapes."
        reflection_directions = flux_centers - concentrator_origins
        return reflection_directions
    
    @staticmethod
    def calc_cosine_similarities(pred_vector: torch.Tensor, true_vector: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
        """
        Cosine similarity between predicted and true direction vectors.

        Parameters
        ----------
        pred_vector, true_vector : torch.Tensor
            Shape [..., 4] or [..., 3] and broadcastable to [B, H, 4]. The last
            component is ignored; vectors are reduced to 3D.
        epsilon : float
            Numerical stabilizer to avoid division by zero.

        Returns
        -------
        torch.Tensor
            Shape [B, H]. Cosine similarities in [-1, 1].
        """
        assert pred_vector.shape == true_vector.shape, 'Given pred_vector and true_vector must have identical shapes.'
        
        # Bring to shape [B, H, 4], if other shape is given
        while pred_vector.dim() < 3:  
            pred_vector = pred_vector.unsqueeze(0)
            true_vector = true_vector.unsqueeze(0) 
        
        # Delete 4th coordinate dimension
        pred_vector = pred_vector[:, :, 0:3]
        true_vector = true_vector[:, :, 0:3]
        
        # Calculate normalized vectors
        m1 = torch.norm(pred_vector, dim=-1)
        m2 = torch.norm(true_vector, dim=-1)
        
        # Calculate cosine-similarity
        dot_products = torch.sum(pred_vector * true_vector, dim=-1)
        cos_sim = (dot_products + epsilon) / (m1 * m2 + epsilon)  # avoid division by zero
        return cos_sim
    
    @staticmethod
    def calc_alignment_errors_stable(pred_vector: torch.Tensor, true_vector: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
        """
        Angular misalignment (mrad) between direction vectors, numerically stable.

        Parameters
        ----------
        pred_vector, true_vector : torch.Tensor
            Shape [..., 4] or [..., 3] and broadcastable to [B, H, 4]. Reduced to 3D.
        epsilon : float
            Numerical stabilizer for robust acos input.

        Returns
        -------
        torch.Tensor
            Shape [B, H]. Angular error in milliradians.
        """
        assert pred_vector.shape == true_vector.shape, 'Given pred_vector and true_vector must have identical shapes.'
        
        # Bring to shape [B, H, 4], if other shape is given
        while pred_vector.dim() < 3:  
            pred_vector = pred_vector.unsqueeze(0)
            true_vector = true_vector.unsqueeze(0) 
        
        # Delete 4th coordinate dimension
        pred_vector = pred_vector[:, :, 0:3]
        true_vector = true_vector[:, :, 0:3]
        
        # Calculate normalized vectors
        m1 = torch.norm(pred_vector, dim=-1, dtype=torch.float64)
        m2 = torch.norm(true_vector, dim=-1, dtype=torch.float64)
        
        # Calculate cosine-similarity
        dot_products = torch.sum(pred_vector * true_vector, dim=-1)
        cos_sim = (dot_products + epsilon) / (m1 * m2 + epsilon)  # avoid division by zero
        
        #  eps_clamp = 1e-12  # Safe clamping with stable edges
        angles_rad = torch.acos(torch.clamp(cos_sim, min= -1.0, max= 1.0))
        if torch.isnan(angles_rad).any():
            log.warning("NaN detected in alignment errors tensor.")
        
        return angles_rad * 1000  # in mrad

    @staticmethod
    def find_upper_contour(bitmaps: torch.Tensor, threshold=0.6, sharpness=70.0, num_interpolate=4, sigma_in=20.0, sigma_out=25.0) -> torch.Tensor:
        """
        Extract smoothed upper contours from flux bitmaps (autograd‑safe).

        Pipeline
        --------
        1) Normalize + interpolate (anti‑alias smoothing).
        2) Gaussian blur (sigma_in).
        3) Soft vertical contour detection (threshold, sharpness).
        4) Gaussian blur (sigma_out) and re‑interpolate + normalize.

        Parameters
        ----------
        bitmaps : torch.Tensor
            Shape [B, He, Hu] or [He, Hu] (broadcasted to [B, ...]).
        threshold : float
            Soft detection threshold in [0, 1].
        sharpness : float
            Slope parameter for the soft transition.
        num_interpolate : int
            Number of up/down interpolation cycles.
        sigma_in, sigma_out : float
            Standard deviations for input/contour Gaussian smoothing.

        Returns
        -------
        torch.Tensor
            Shape [B, He, Hu]. Upper‑contour bitmaps in [0, 1].
        """
        # 1. Normalize and interpolate for smoothing
        norm_bitmaps = normalize_and_interpolate(bitmaps, num_interpolate).squeeze(0)  # [B, H, W]
        
        smoothed_contours = []
        for b in range(norm_bitmaps.shape[0]):
            img = norm_bitmaps[b] # [H, W]
            # 2. Add Gaussian blur to input
            gauss_img = gaussian_filter_2d(img, sigma=sigma_in)  # [H, W]
            norm_gauss_img = normalize_images(gauss_img.unsqueeze(0))
            # 3. Search for upper contour based on hyper-parameters
            contour = find_soft_contour_pytorch_vertical(norm_gauss_img, threshold=threshold, sharpness=sharpness)  # [1, 1, H, W]
            # 4. Add Gaussian blur to contour for smoothing
            smoothed = gaussian_filter_2d(contour, sigma=sigma_out).squeeze(0)  # [H, W]
            smoothed_contours.append(smoothed)

        output = torch.stack(smoothed_contours, dim=0)  # [B, H, W]
        interpolated = normalize_and_interpolate(output, num_interpolate).squeeze(0)
        return interpolated
    
    def count_empty_flux_predictions(self, pred_flux_bitmaps: torch.Tensor, epoch: int, threshold=1) -> Tuple[torch.Tensor, int]:
        """
        Count and flag empty predicted flux bitmaps.

        Parameters
        ----------
        pred_flux_bitmaps : torch.Tensor
            Shape [B, H, He, Hu]. Assumed normalized to [0, 1].
        epoch : int
            Current epoch for logging.
        threshold : float
            Minimum sum of pixel intensities to be considered non‑empty.

        Returns
        -------
        empty_mask : torch.Tensor
            Shape [B, H], boolean. True where a heliostat’s flux image is empty.
        empty_count : int
            Total number of empty (B, H) instances.
        """
        assert pred_flux_bitmaps.dim() > 2, \
            "Expected pred_flux_bitmaps to have at least 3 dimensions, not {pred_flux_bitmaps.dim()} dimensions."
        sums = pred_flux_bitmaps.sum(dim=[-1, -2])
        min_sum = sums.min()
        empty_mask = (sums < threshold)  # [B, H]
        empty_count = empty_mask.sum().item()
        self.tb_logger.log_metric(epoch, "CountOfEmptyFluxPredictions", empty_count)
        
        if empty_count > 0:
            log_steps = self.run_config[my_config_dict.run_config_general][my_config_dict.general_logging_steps]
            if epoch % log_steps == 0:
                log.info(f"{self.tb_logger.mode} Batch: Count of empty flux bitmap predictions: {empty_count} of {empty_mask.numel()}")
        return empty_mask, empty_count
    
    def freeze_params(self, freeze_params: List[str] = [],  logging: bool=True) -> None:
        """
        Freeze or unfreeze parameter groups in `self.learnable_params_dict`.

        Parameters
        ----------
        freeze_params : list[str]
            Collection of group name substrings to freeze. All others are unfrozen.
        logging : bool
            If True, print which groups are frozen.
        """
        # Iterate over the nested parameters and freeze / unfreeze
        # Names and parameter lists in the learnable parameters dict
        for name, param_group in self.learnable_params_dict.items():
            grad = True  
            # Check if the parameter group 
            if name in freeze_params:
                if logging:
                    log.info(f"Freezing parameters '{name}'.") 
                grad = False
            for params in param_group:
                if isinstance(params, (torch.nn.ParameterList, List, Tuple)):
                    for param in params:
                        param.requires_grad = grad
                        has_grad = param.requires_grad    
                else:
                    params.requires_grad = grad
                    has_grad = params.requires_grad       
                    
    def unfreeze_param_group(self, param_group: List[str]) -> None:
        """
        Unfreeze only parameters whose names match any substring in `param_group`.

        All other groups are frozen (except those permanently frozen in config).

        Parameters
        ----------
        param_group : list[str]
            Substrings to select the group(s) to unfreeze for the current phase.
        """
        model_params = list(self.learnable_params_dict.keys())
        # Never unfreeze selection of permanently frozen params
        freeze_and_phase = self.run_config[my_config_dict.run_config_model][my_config_dict.model_apply_with_caution]
        freeze_params = set(freeze_and_phase[my_config_dict.model_freeze_params])
        # Find parameters that will be frozen and add to freeze_parameters
        for param_key in param_group:
            no_match = set(param for param in model_params if not (param_key in param))
            # Join no_match (params that are excluded from current phase -> freeze them)
            freeze_params = freeze_params | no_match
        
        self.freeze_params(freeze_params=list(freeze_params), logging=False)
    
    def phase_freeze_training(self, current_epoch: int, phase_counter: int) -> int:
        """
        Phase‑based unfreezing according to config‑defined epochs and phase lengths.

        Parameters
        ----------
        current_epoch : int
            Global epoch index (resets internal phase counter at phase starts).
        phase_counter : int
            Rolling counter to select the next parameter group to unfreeze.

        Returns
        -------
        int
            Updated `phase_counter` for the calling training loop.
        """
        epochs_sequence = self.run_config[my_config_dict.run_config_model][my_config_dict.model_epochs_sequence]
        freeze_and_phase = self.run_config[my_config_dict.run_config_model][my_config_dict.model_apply_with_caution]
        phase_lengths_sequence = freeze_and_phase[my_config_dict.model_length_of_phases]
        assert len(epochs_sequence) == len(phase_lengths_sequence), \
            "Length of epochs_sequence must be equal to phase_lengths_sequence. Please change the config-file."
        
        phase_param_groups = freeze_and_phase[my_config_dict.model_param_groups_keywords]
        num_param_groups = len(phase_param_groups)
        
        epoch_accumulator = 0
        for (epochs, phase_length) in zip(epochs_sequence, phase_lengths_sequence):
            # Determine the length of the current phase of epochs (sequence lr scheduler)
            if current_epoch < epoch_accumulator + epochs:
                # How far into the current LR phase we are
                local_epoch = current_epoch - epoch_accumulator
                
                if current_epoch == epoch_accumulator:
                    phase_counter = 0
                    
                # Chek if we enter a new phase which leads to unfreezing respective parameters
                if local_epoch % phase_length == 0:
                    # Calculate index to determine which group of parameters will be unfrozen
                    phase_group_index = phase_counter % num_param_groups 
                    log.info(f"Entering Phased Learning for group {phase_param_groups[phase_group_index]}")
                    self.unfreeze_param_group(phase_param_groups[phase_group_index])
                    phase_counter += 1

                return phase_counter
            
            # Move to next epoch sequence in lr scheduler
            epoch_accumulator += epochs

    def update_best_parameters(self, best_metric: float, current_metric: torch.Tensor, best_state_dict: dict()) -> Tuple[float, dict]:
        """
        Track and store the best model state according to a validation metric.

        Parameters
        ----------
        best_metric : float
            Best (lowest) metric value seen so far.
        current_metric : torch.Tensor
            Current (scalar) validation metric (lower is better).
        best_state_dict : dict
            Previously saved best `state_dict()`.

        Returns
        -------
        best_metric : float
        best_state_dict : dict
            Updated best metric and a deep copy of the current `state_dict` when
            improvement is observed; otherwise unchanged.

        """
        if not torch.isnan(current_metric) and (current_metric.item() < best_metric):
            best_metric = current_metric.item()
            best_state_dict = copy.deepcopy(self.state_dict())
        return best_metric, best_state_dict

    def load_best_parameters(self, best_state_dict) -> None:
        """
        Load a previously captured best `state_dict` into the model (if any).

        Parameters
        ----------
        best_state_dict : dict or None
            Saved parameter dictionary. If None, the call is a no‑op.

        """
        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
