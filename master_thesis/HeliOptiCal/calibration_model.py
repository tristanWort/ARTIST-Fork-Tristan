import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity 
from torch.utils.tensorboard import SummaryWriter 
# import tensorflow.summary as summary
import logging
import sys
import os
import re
import json

from datetime import datetime
from torchmetrics.image import StructuralSimilarityIndexMeasure
from typing import Union, Literal, List, Tuple, Dict, Optional
from pathlib import Path

# parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if parent_path not in sys.path:
#     sys.path.append(parent_path) 

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 
from artist.util.scenario import Scenario
from artist.raytracing import raytracing_utils
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util import utils

import my_config_dict
from calibration_datasplitter import CalibrationDataSplitter
from calibration_dataloader import CalibrationDataLoader
from logger import TensorboardLogger
from util import (get_rigid_body_kinematic_parameters_from_scenario, 
                  count_parameters, 
                  check_for_nan_grad, 
                  create_parameter_groups, 
                  find_soft_contour_vertical,
                  normalize_and_interpolate)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
# A logger for the CalibrationModel.


class CalibrationModel(nn.Module):
    """
    A class to represent a Heliostat Calibration Model.

    Attributes
    ----------
    name : str
        Name used for storing model data.

    Methods
    -------

    """
    def __init__(self, run_config: Dict[str, Dict]):
        """
        Inialize a Heliostat Calibration Model.
        
        Parameters
        ----------
        config : Dict
            The run-config dictionary for a Heliostat Calibration Model.
        """        
        super(CalibrationModel, self).__init__()
        
        # Perform general model configurations
        self.run_config = run_config  # save for later usage
        general_config = run_config[my_config_dict.run_config_general]
        
        self.name = f"run_{datetime.now().strftime('%y%m%d%H%M')}_{general_config[my_config_dict.general_name]}" 
        log.info(f"Initializing a Heliostat Calibration Model: {self.name}")
        self.save_directory = Path(general_config[my_config_dict.general_save_to_directory]) / self.name
        self.tb_logger = None  # later set logger for present run
        
        self.device = general_config[my_config_dict.general_device]
        
        # Load the scenario
        self.scenario, self.target_areas = self.load_scenario(general_config[my_config_dict.general_scenario_path])
        self.heliostat_ids = self.scenario.heliostat_field.all_heliostat_names
        
        # Initiate Raytracer
        self.use_raytracing = bool(general_config[my_config_dict.general_use_raytracing])
        self.raytracer = self.init_raytraycer(self.scenario)
        log.info("Done with Scenario loading and HeliostatRayTracer setup.")
        
        # Perform configurations for model
        model_config = run_config[my_config_dict.run_config_model]
        self.learnable_params_dict, parameter_count, element_count = (get_rigid_body_kinematic_parameters_from_scenario(
                kinematic=self.scenario.heliostat_field.rigid_body_kinematic))
        log.info(f"Found number of parameters and elements: {parameter_count}, {element_count}")
        
        self.optimizer = self.configure_optimizer(model_config)
        self.schedulers = self.configure_schedulers(model_config)
        log.info("Done with optimizer and scheduler setup.")
        
        self.datasplitter, self.dataloader = self.configure_dataloader(run_config[my_config_dict.run_config_dataset])
        log.info("Done with data loading data splits.")
        json.dump(run_config, open(self.save_directory / 'run_config.json', 'w+'), indent=4)
        
    def load_scenario(self, scenario_path: Union[str, Path]):
        """
        Load and return an `ARTIST` Scenario.
        """
        # TODO: Manipulate scenario if errors are configured
        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Scenario file not found at path: {scenario_path}")

        import h5py
        # Load the scenario from the h5-file.
        with h5py.File(scenario_path, "r") as scenario_file:
            loaded_scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file, 
                device=self.device
            )

        # Add errors to kinematic model
        if bool(self.run_config[my_config_dict.run_config_general][my_config_dict.general_introduce_random_errors]):
            from add_random_errors import add_random_errors_to_kinematic
            error_config = self.run_config[my_config_dict.run_config_initial_errors]
            seed = self.run_config[my_config_dict.run_config_general][my_config_dict.general_random_seed]            
            error_kinematic = add_random_errors_to_kinematic(error_config, loaded_scenario, self.save_directory/'errors',
                                           seed=seed, device=self.device)
            loaded_scenario.heliostat_field.rigid_body_kinematic = error_kinematic
            log.info("Added initial errors to Heliostat field kinematic.")

        target_areas = {area.name: area for area in loaded_scenario.target_areas.target_area_list}
        return loaded_scenario, target_areas
    
    def init_raytraycer(self, scenario: Scenario, world_size=1, rank=0, batch_size=1, random_seed=42, bitmap_resolution=(256, 256)):
        """
        Iniitiate and return a HeliostatRayTracer.
        """
        # Get the loss type configurations 
        loss_config = self.run_config[my_config_dict.run_config_general][my_config_dict.general_loss_config]
        model_flux_center_predictions = loss_config[my_config_dict.general_mode_flux_center].lower()
        
        # Perform sanity check for the given configurations for raytracer use and loss type
        if self.use_raytracing and ('ideal' in model_flux_center_predictions):
            log.warning(f"Raytracer use is configured although your chosen loss only considers Heliostat orientations."\
                "Using the Raytracer leads to increased runtimes. Make sure that this is what you want.")
        elif not self.use_raytracing and not ('ideal' in model_flux_center_predictions):
            raise TypeError(f"Your chosen loss type is image based and therefore requires Raytracer use to be activated."\
                "Please change your configurations.")
            
        # Before raytracer setup, Heliostat field needs to be aligned
        heliostat_field = scenario.heliostat_field
        heliostat_field.align_surfaces_with_incident_ray_direction(
            # initial alignment (direction does not matter)
            incident_ray_direction = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device),
            device=self.device
            )
        
        raytracer = HeliostatRayTracer(scenario=scenario,
                                       world_size=world_size,
                                       rank=rank,
                                       batch_size=batch_size,
                                       random_seed=random_seed,
                                       bitmap_resolution_e=bitmap_resolution[0],
                                       bitmap_resolution_u=bitmap_resolution[1])
        return raytracer
    
    def configure_optimizer(self, model_config: Dict):
        """
        Configure optimizer using the config dictionary.
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
    
    def configure_schedulers(self, model_config: Dict):
        """
        Configure and return a learning rate scheduler.
        If multiple schedulers are defined, returns a SequentialLR.
        If one scheduler is defined, returns it directly.
        """
        optimizer = self.optimizer
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
                    step_size_up=[my_config_dict.model_cycliclr_step_size_up],
                    mode=cfg[my_config_dict.model_cycliclr_mode],
                    gamma=cfg[my_config_dict.model_cycliclr_gamma],
                    cycle_momentum=False
                )

            elif scheduler_type == 'ReduceLROnPlateau':
                cfg = model_config[my_config_dict.model_reduce_on_plateau]
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=cfg[my_config_dict.model_reduce_on_plateau_factor],
                    patience=cfg[my_config_dict.model_reduce_on_plateau_patience],
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
        
    def configure_dataloader(self, dataset_config: Dict):
        """
        
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
        # Configure dataloader
        calibration_data_loader = CalibrationDataLoader(
            data_directory=dataset_config[my_config_dict.dataset_training_data_directory],
            heliostats_to_load=self.heliostat_ids,
            power_plant_position=self.scenario.power_plant_position,
            load_flux_images=True, #bool(self.use_raytracing),
            preload_flux_images=bool(dataset_config[my_config_dict.dataset_preload_flux_images]),
            is_simulated_data=bool(dataset_config[my_config_dict.dataset_training_data_was_simulated]),
            properties_file_ends_with=dataset_config[my_config_dict.dataset_properties_suffix],
            flux_file_ends_with=dataset_config[my_config_dict.dataset_flux_suffix],
            device=dataset_config[my_config_dict.dataset_device]
        )
        return data_splitter, calibration_data_loader
        
    def forward(self, field_batch: List[Dict[str, Union[torch.Tensor, str]]], device: Optional[Union[torch.device, str]] = None):
        """
        Generate model output based on the given input data.
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
            
            # Perform Raytracing if configured
            if self.use_raytracing:
                incident_ray_directions = data[my_config_dict.field_sample_incident_rays]
                target_area_names = data[my_config_dict.field_sample_target_names]
                target_areas = [self.target_areas[name] for name in target_area_names]
                pred_flux_bitmaps[sample] = raytracer.trace_rays_separate(incident_ray_directions=incident_ray_directions,
                                                                    target_areas=target_areas,
                                                                    device=device)
                
                # Normalization caused CUDA out of memory error
                # pred_flux_bitmaps[sample] = self.raytracer.normalize_bitmaps(bitmaps=sample_flux_bitmaps, target_areas=target_areas)
                
        return field_orientations, pred_flux_bitmaps
    
    @staticmethod
    def calc_reflection_directions_from_flux_centers(flux_centers: torch.Tensor, orientations: torch.Tensor):
        # Add dimensions batchsize and Heliostat index if not given
        while (d := flux_centers.dim()) < 3: flux_centers = flux_centers.unsqueeze(0) # to [B, H, 4] 
        while (d := orientations.dim()) < 4: orientations = orientations.unsqueeze(0) # to [B, H, 4, 4] 

        # Origin coordinates can be extracted from orientations
        concentrator_origins = orientations[:, :, 0:4, 3]
        assert flux_centers.shape == concentrator_origins.shape, \
            "Given flux_centers and concentrator_origins should have equal shapes."
        reflection_directions = flux_centers - concentrator_origins
        return reflection_directions
    
    def calc_flux_centers_from_orientations(self, 
                                            orientations: torch.Tensor, 
                                            incident_ray_directions: torch.Tensor,
                                            target_area_names: List[List[str]]):
        from util import calculate_intersection
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
    def calc_reflection_directions_from_orientations(incident_ray_directions: torch.Tensor, orientations: torch.Tensor):
        """
        Calculate reflected ray directions using incident ray directions and concentrator orientations.

        Parameters
        ----------
        incident_ray_directions : torch.Tensor
            Tensor of shape [B, H, 4] containing the direction of incoming rays.
        orientations : torch.Tensor
            Tensor of shape [B, H, 4, 4] representing the orientation matrices of heliostats.

        Returns
        -------
        torch.Tensor
            Reflected ray directions of shape [B, H, 4], normalized and in ENU 4D format.
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

        # Optional: normalize result (already unit length if inputs are)
        return reflected
    
    def calc_centers_of_mass(self, bitmaps: torch.Tensor, target_area_names: List[List[str]], threshold=0):
        """
        Calculate and return the ENU-coordinates of the center of mass for a batch of bitmaps.
        """
        from artist.util.utils import get_center_of_mass
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
    
    def calc_pred_reflection_directions(self, pred_flux_bitmaps: torch.Tensor, orientations: torch.Tensor):
        """
        
        """
        # Add dimensions batchsize and Heliostat index if not given
        while (d := pred_flux_bitmaps.dim()) < 4: pred_flux_bitmaps = pred_flux_bitmaps.unsqueeze(0) # to [B, H, (res)] 
        while (d := orientations.dim()) < 4: orientations = orientations.unsqueeze(0) # to [B, H, 4, 4] 
        
        if not self.use_raytracing: 
            # Calculate reflection direction using only the orientations matrix
            concentrator_normal = orientations    
        else:
            assert pred_flux_centers.shape[:2] == orientations.shape[:2], \
                "pred_flux_centers and orientations must have equal batch sizes and number of Heliostat."
    
    @staticmethod
    def extract_upper_contours(bitmaps: torch.Tensor, threshold=0.5, sharpness=30.0, num_interpolate=0):
        """
        
        """
        bitmaps = normalize_and_interpolate(bitmaps, num_interpolate, 2, bitmaps.shape[-1])            
        upper_contour_bitmaps = find_soft_contour_vertical(bitmaps, threshold, sharpness)
        upper_contour_bitmaps = normalize_and_interpolate(upper_contour_bitmaps, num_interpolate, 2, bitmaps.shape[-1])
        return upper_contour_bitmaps

    def evaluate_model(self, epoch: int, orientations: torch.Tensor, pred_flux_bitmaps: torch.Tensor, field_batch: List[Dict]):
        """
        
        """
        # TODO: Normalize true and predicted Bitmaps!
        # Get nested list of target area names and stacked Tensor of incident ray directions from batch data
        target_areas = [sample[my_config_dict.field_sample_target_names] for sample in field_batch]
        incident_ray_directions = torch.stack([sample[my_config_dict.field_sample_incident_rays] for sample in field_batch])
        # true_flux_centers = torch.stack([sample[my_config_dict.field_sample_flux_centers] for sample in field_batch])
        
        true_flux_bitmaps = torch.stack([sample[my_config_dict.field_sample_flux_images] for sample in field_batch])
        with torch.no_grad():
            true_flux_centers = self.calc_centers_of_mass(bitmaps=true_flux_bitmaps, target_area_names=target_areas)
        
        # All inputs must have the same batch size
        assert len(target_areas) == orientations.shape[0] == pred_flux_bitmaps.shape[0],\
            "Batch sizes must be equal for target_areas, orientations and pred_flux_bitmaps."
        
        # Get the loss type configurations 
        loss_config = self.run_config[my_config_dict.run_config_general][my_config_dict.general_loss_config]
        mode = loss_config[my_config_dict.general_mode_flux_center].lower()
        
        # Mask for empty bitmaps
        if self.use_raytracing:
            sums = pred_flux_bitmaps.sum(dim=[-1, -2])
            empty_mask = (sums < 0.001)  # [B, H]
            empty_count = empty_mask.sum().item()
            self.tb_logger.log_metric(epoch, "EmptyPredFluxCount", empty_count)
            if empty_count > 0:
                log_steps = self.run_config[my_config_dict.run_config_general][my_config_dict.general_logging_steps]
                if epoch % log_steps == 0:
                    log.info(f"{self.tb_logger.mode} Batch: Count of empty flux predictions: {empty_count} of {empty_mask.numel()}")

        if "ideal" in mode:
            # Calculate the reflcetion directions directly form the Heliostat orientations
            pred_reflection_directions = self.calc_reflection_directions_from_orientations(
                incident_ray_directions=incident_ray_directions,
                orientations=orientations)
        
        else:  # use flux bitmaps
        
            # Fallback reflection directions (for empty bitmaps)
            fallback_reflections = self.calc_reflection_directions_from_orientations(incident_ray_directions=incident_ray_directions,
                                                                                     orientations=orientations)
                
            if "measured" in mode:
                # Calculate the center of mass coordinates of focal spots
                pred_flux_centers = self.calc_centers_of_mass(bitmaps=pred_flux_bitmaps, target_area_names=target_areas)
            
            elif "contour" in mode:               
                # Extracts upper contour bitmaps of both the true and predicted flux images
                true_contour_bitmaps = self.extract_upper_contours(bitmaps=true_flux_bitmaps, threshold=0.5, num_interpolate=4)
                pred_contour_bitmaps = self.extract_upper_contours(bitmaps=pred_flux_bitmaps, threshold=0.5, num_interpolate=4)
                
                diff_contour_bitmaps = pred_contour_bitmaps - true_contour_bitmaps
                self.tb_logger.log_flux_bitmaps(epoch=epoch, bitmaps=true_contour_bitmaps, type='TrueContour')
                self.tb_logger.log_flux_bitmaps(epoch=epoch, bitmaps=pred_contour_bitmaps, type='PredContour')
                self.tb_logger.log_flux_bitmaps(epoch=epoch, bitmaps=diff_contour_bitmaps, type='DiffContour')
                # Use coordinates of contour centers for alignment
                true_flux_centers = self.calc_centers_of_mass(bitmaps=true_contour_bitmaps, target_area_names=target_areas)
                pred_flux_centers = self.calc_centers_of_mass(bitmaps=pred_contour_bitmaps, target_area_names=target_areas)
                
            else:
                raise ValueError(f"Unsupported mode for flux center predictions: {model_flux_center_predictions}")

            # Use flux centers where available, otherwise use fallback
            reflection_from_centers = self.calc_reflection_directions_from_flux_centers(pred_flux_centers, orientations)
            pred_reflection_directions = torch.where(empty_mask.unsqueeze(-1),  # [B, H, 1]
                                                    fallback_reflections,
                                                    reflection_from_centers)
            
        # Prepare loss on alignment errors
        true_reflection_directions = self.calc_reflection_directions_from_flux_centers(flux_centers=true_flux_centers, orientations=orientations)
        cosine_similarites = self.calc_cosine_similarities(pred_vector=pred_reflection_directions, true_vector=true_reflection_directions)
        # Calculate loss on cosine similarities
        loss = self.calc_loss_from_cosine_similarities(cosine_similarities=cosine_similarites, epoch=epoch)
                
        # Calculate the evaluation metric (alignment error)
        with torch.no_grad():
            if self.dataloader.is_simulated_data:  # data should contain the ideal flux centers
                true_flux_centers = torch.stack([sample[my_config_dict.field_sample_ideal_flux_centers] for sample in field_batch])
                
            else:  # use measured centers
                # true_flux_centers = torch.stack([sample[my_config_dict.field_sample_flux_centers] for sample in field_batch])
                true_flux_centers = self.calc_centers_of_mass(bitmaps=true_flux_bitmaps, target_area_names=target_areas)
            
            true_reflection_directions = self.calc_reflection_directions_from_flux_centers(flux_centers=true_flux_centers,
                                                                                            orientations=orientations)     
            # Calculate alignment errors for model evaluation
            alignment_errors = self.calc_alignment_errors_stable(pred_vector=pred_reflection_directions, true_vector=true_reflection_directions)
        
        self.tb_logger.log_alignment_errors(epoch=epoch, alignment_errors=alignment_errors)
        
        if self.use_raytracing:
            self.tb_logger.log_flux_bitmaps(epoch=epoch, bitmaps=true_flux_bitmaps, type='TrueFlux')
            self.tb_logger.log_flux_bitmaps(epoch=epoch, bitmaps=pred_flux_bitmaps, type='PredFlux')
            diff_flux_bitmaps = pred_flux_bitmaps - true_flux_bitmaps
            self.tb_logger.log_flux_bitmaps(epoch=epoch, bitmaps=diff_flux_bitmaps, type='DiffFlux')
            
        return loss, alignment_errors         

    def calibrate(self, device: Optional[Union[torch.device, str]] = None):
        """
        
        """ 
        device = torch.device(self.device if device is None else device)
        # Sum of scheduled epochs
        sum_epochs = sum(self.run_config[my_config_dict.run_config_model][my_config_dict.model_epochs_sequence])
        log_steps = self.run_config[my_config_dict.run_config_general][my_config_dict.general_logging_steps]
        tolerance = self.run_config[my_config_dict.run_config_general][my_config_dict.general_tolerance_mrad]
        
        splits = self.datasplitter.splits
        for split_type in splits:
            for split_size, split_df in splits[split_type].items():
                log.info(f"Begin calibration for dataset split {split_type}: size {split_size}")
                
                run = f"{split_type}_{split_size}"
                # Reset Logger, Scenario, Raytracer, and Parameters
                # self.scenario, self.target_areas = self.load_scenario(self.run_config[my_config_dict.run_config_general][my_config_dict.general_scenario_path])
                # self.learnable_params_dict, _, _ = get_rigid_body_kinematic_parameters_from_scenario(self.scenario.heliostat_field.rigid_body_kinematic)
                # self.raytracer = self.init_raytraycer(self.scenario)
                self.tb_logger = TensorboardLogger(run=run, heliostat_names=self.heliostat_ids, log_dir=self.save_directory / 'logs')
                
                # Set gradients to False for selection of frozen parameters
                freeze_and_phase = self.run_config[my_config_dict.run_config_model][my_config_dict.model_apply_with_caution]
                freeze_params = freeze_and_phase[my_config_dict.model_freeze_params]
                if len(freeze_params) > 0:
                    self.freeze_params(freeze_params)  # freeze params for this run
                    
                # Check, whether to perform phased learning / training
                phased_learning_bool = bool(freeze_and_phase[my_config_dict.model_phase_param_learning])
                phase_counter = 0  # required to track the phases for learning
                               
                epoch = 0
                best_val_error = float('inf')
                worst_error_across_field = float('inf')
                
                # Set early stopping if worst error drops below tolerance
                while epoch < sum_epochs and worst_error_across_field > tolerance:
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
                    train_loss, train_alignment_errors = self.evaluate_model(epoch, field_orientations, pred_flux_bitmaps, data_batch_train)    
                    self.log_alignment_errors(epoch=epoch, alignment_errors=train_alignment_errors, helio_and_calib_ids=split_ids_train)              

                    # Log parameters
                    self.tb_logger.log_parameters_obj(epoch, self.learnable_params_dict)
                
                    # Backward
                    self.optimizer.zero_grad()
                    train_loss.backward()
                    check_for_nan_grad(self.learnable_params_dict)
                    self.optimizer.step()
                        
                    # Validation
                    self.eval()
                    split_ids_val = self.datasplitter.get_helio_and_calib_ids_from_split(split_type, split_size, 'validation', heliostat_ids=self.heliostat_ids)
                    self.tb_logger.set_mode("Valid")
                    self.tb_logger.set_helio_and_calib_ids(split_ids_val)
                    with torch.no_grad():
                        data_batch_val = self.dataloader.get_field_batch(helio_and_calib_ids=split_ids_val, device=device)
                        val_orientations, val_pred_bitmaps = self.forward(data_batch_val)
                        val_loss, val_alignment_errors = self.evaluate_model(epoch, val_orientations, val_pred_bitmaps, data_batch_val)
                    self.log_alignment_errors(epoch=epoch, alignment_errors=val_alignment_errors, helio_and_calib_ids=split_ids_val) 
                    
                    # Step schedulers if not None
                    if self.schedulers:
                        if isinstance(self.schedulers, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.schedulers.step(val_loss)
                            current_lr = current_lr = self.optimizer.param_groups[0]['lr']
                        else:
                            self.schedulers.step()
                            current_lr = self.schedulers.get_last_lr()[0]
                    else:
                        current_lr = current_lr = self.optimizer.param_groups[0]['lr']
                    self.tb_logger.log_metric(epoch, 'BaseLearningRate', current_lr)
                    
                    epoch += 1
                    
                    
                    if epoch % log_steps == 0:
                        
                        log.info(f"Epoch: {epoch} of {sum_epochs}, "
                         f"Train / Val Loss: {train_loss.item():.6f} / {val_loss.item():.6f}, "
                         f"Train / Val Avg.Error (mrad): {train_alignment_errors.mean().item():.2f} / {val_alignment_errors.mean().item():.2f}, "
                         f"LR: {current_lr}")
                    
                    if val_alignment_errors.max().item() < worst_error_across_field:
                        worst_error_across_field = val_alignment_errors.max().item()
                    
                    
                # Save last parameters
                self.tb_logger.log_parameters_obj(epoch, self.learnable_params_dict)
                
                # Testing
                self.eval()
                split_ids_test = self.datasplitter.get_helio_and_calib_ids_from_split(split_type, split_size, 'test', heliostat_ids=self.heliostat_ids)
                self.tb_logger.set_mode("Test")
                self.tb_logger.set_helio_and_calib_ids(split_ids_test)
                data_batch_test = self.dataloader.get_field_batch(helio_and_calib_ids=split_ids_test, device=device)
                test_orientations, test_pred_bitmaps = self.forward(data_batch_test)
                _, test_alignment_errors = self.evaluate_model(epoch, test_orientations, test_pred_bitmaps, data_batch_test)
                self.log_alignment_errors(epoch=epoch, alignment_errors=test_alignment_errors, helio_and_calib_ids=split_ids_test)
                log.info(f"Finally: Test Avg.Error (mrad): {test_alignment_errors.mean().item():.2f}") 
            
    def phase_freeze_training(self, current_epoch: int, phase_counter: int):
        """
        Method uses phased training, if configured.
        The selection of frozen parameters will not be affected by phased training.
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
            
    def freeze_params(self, freeze_params: List[str] = [], logging: bool=True):
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
                    
    def unfreeze_param_group(self, param_group: List[str]):
        """
        Unfreeze current parameter group, while freezing all other parameters.
        
        Parameters
        ----------
        param_group : List[str]
            List of key-words for selecting those parameters which will be unfrozen for the current phase.
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
    
    @staticmethod
    def calc_cosine_similarities(pred_vector: torch.Tensor, true_vector: torch.Tensor, epsilon: float = 1e-10): 
        assert pred_vector.shape == true_vector.shape, 'Given pred_vector and true_vector must have identical shapes.'
        
        # Bring to shape [B, H, 4], if other shape is given
        while pred_vector.dim() < 3:  
            pred_vector = pred_vector.unsqueeze(0)
            true_vector = true_vector.unsqueeze(0) 
        
        # Calculate normalized vectors
        m1 = torch.norm(pred_vector, dim=-1, dtype=torch.float64)
        m2 = torch.norm(true_vector, dim=-1, dtype=torch.float64)
        
        # Calculate cosine-similarity
        dot_products = torch.sum(pred_vector * true_vector, dim=-1)
        cos_sim = dot_products / (m1 * m2 + epsilon)  # avoid division by zero
        return cos_sim
    
                
    def calc_loss_from_cosine_similarities(self, cosine_similarities: torch.Tensor, epoch: int):
        """
        Calculate loss from cosine similarities for each heliostat independently.

        Args:
            cosine_similarities (torch.Tensor): Tensor of shape [B, H_idx], values in [-1.0, 1.0]

        Returns:
            torch.Tensor: Tensor with mean loss value for Heliostat field
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
            per_heliostat_loss = torch.mean(torch.abs(cosine_similarities - target), dim=0)  # [H_idx]

        # L2 / MSE Loss per heliostat
        elif any(key in loss_type for key in ['MSE', 'L2']):
            per_heliostat_loss = torch.mean((cosine_similarities - target) ** 2, dim=0)  # [H_idx]

        # MAX error (L∞ loss) per heliostat
        elif 'MAX' in loss_type:
            per_heliostat_loss = torch.max(torch.abs(cosine_similarities - target), dim=0).values  # [H_idx]
        
        # Huber implementation with tunable delta
        elif 'HUBER' in loss_type:
            # Extract delta from string using regex, fallback to default if not found
            match = re.search(r'HUBER[-_]?(\d*\.?\d+)', loss_type)
            delta = float(match.group(1)) if match else 0.00005  # default ≈ 10 mrad

            diff = cosine_similarities - target
            abs_diff = torch.abs(diff)
            huber = torch.where(
                abs_diff < delta,
                0.5 * diff**2,
                delta * (abs_diff - 0.5 * delta)
            )
            per_heliostat_loss = huber.mean(dim=0)

        # Quantile or Percentile implementation with tunable value
        elif 'QUANTILE' in loss_type or 'PERCENTILE' in loss_type:
            match = re.search(r'(QUANTILE|PERCENTILE)[-_]?(\d+)', loss_type)
            quantile_val = float(match.group(2)) if match else 90.0  # default 90
            quantile = quantile_val / 100.0
            per_heliostat_loss = torch.quantile(torch.abs(cosine_similarities - target), q=quantile, dim=0)

        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        loss = per_heliostat_loss.mean()
        self.tb_logger.log_loss(f"Loss_{loss_type}", loss.item(), epoch)
        
        return per_heliostat_loss.mean() * sf_loss
        
    @staticmethod
    def calc_alignment_errors_stable(pred_vector: torch.Tensor, true_vector: torch.Tensor, epsilon: float = 1e-10): 
        assert pred_vector.shape == true_vector.shape, 'Given pred_vector and true_vector must have identical shapes.'
        
        # Bring to shape [B, H, 4], if other shape is given
        while pred_vector.dim() < 3:  
            pred_vector = pred_vector.unsqueeze(0)
            true_vector = true_vector.unsqueeze(0) 
        
        # Calculate normalized vectors
        m1 = torch.norm(pred_vector, dim=-1, dtype=torch.float64)
        m2 = torch.norm(true_vector, dim=-1, dtype=torch.float64)
        
        # Calculate cosine-similarity
        dot_products = torch.sum(pred_vector * true_vector, dim=-1)
        cos_sim = dot_products / (m1 * m2 + epsilon)  # avoid division by zero
        
        eps_clamp = 1e-8  # Safe clamping with stable edges
        angles_rad = torch.acos(torch.clamp(cos_sim, min= -1.0 + eps_clamp, max= 1.0 - eps_clamp))
        angles_mrad = angles_rad * 1000
        
        return angles_mrad
    
    def log_parameters(self, epoch: int, obj=None, name=None, heliostat_idx=None, index=None):
        # Delete, instead use TensorboardLogger module
        """
        Log model parameters recursively. To log all paramters, simply call this method once, specifying the current epoch.
        """
        if obj is None:
            # Initiate obj for subsequent recursions
            obj = self.learnable_params_dict  # of type ParameterDict
        else:
            # Use obj from previous recursion
            if isinstance(obj, torch.nn.Parameter):
                if heliostat_idx is None:
                    raise ValueError(f"Missing Heliostat index!")
                heliostat_name = self.heliostat_ids[heliostat_idx]
                self.tb_logger.log_parameter_as_scalar(name, heliostat_name, obj, epoch, index)
            
            elif isinstance(obj, (torch.nn.ParameterList, List, Tuple)):
                if heliostat_idx is None:
                    if len(obj) == len(self.heliostat_ids):
                        for h, param in enumerate(obj):
                            self.log_parameters(epoch=epoch, obj=param, name=name, heliostat_idx=h)
                    else:
                        for i, param in enumerate(obj):
                            self.log_parameters(epoch=epoch, obj=param, name=name, heliostat_idx=h, index=i)
                    
                else: 
                    [self.log_parameters(epoch=epoch, obj=param, name=name, heliostat_idx=heliostat_idx, index=i) 
                        for i, param in enumerate(obj)]

        if isinstance(obj, (torch.nn.ParameterDict, dict)):
            for name, obj in obj.items():
                self.log_parameters(epoch=epoch, obj=obj, name=name)
                
    def log_alignment_errors(self, epoch: int, alignment_errors: torch.Tensor, helio_and_calib_ids: Dict[str, List]):
        # Delete, instead use TensorboardLogger module
        self.tb_logger.log_metric(epoch, "AlignmentErrors_mrad/Avg", alignment_errors.mean().item())
        self.tb_logger.log_metric(epoch, "AlignmentErrors_mrad/Med", torch.median(alignment_errors.flatten()).item())

        for n_helio, (helio_id, calib_ids) in enumerate(helio_and_calib_ids.items()):
            # Log averages of errors per Heliostat
            avg_helio = alignment_errors[:, n_helio].mean()
            med_helio = alignment_errors[:, n_helio].median()
            self.tb_logger.log_heliostat_metric(epoch, "AlignmentErrors_mrad/Avg", helio_id, avg_helio.item())
            self.tb_logger.log_heliostat_metric(epoch, "AlignmentErrors_mrad/Med", helio_id, med_helio.item())
            
            # Log individual errors
            for n_sample, (calib_id, error) in enumerate(zip(calib_ids, alignment_errors[:, n_helio].tolist())):
                self.tb_logger.log_heliostat_metric(epoch, "AlignmentErrors_mrad", helio_id, error, calib_id)
    