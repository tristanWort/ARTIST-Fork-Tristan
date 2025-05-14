import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity 
from torch.utils.tensorboard import SummaryWriter 
# import tensorflow.summary as summary
import logging
import sys
import os
import copy

from torchmetrics.image import StructuralSimilarityIndexMeasure
from typing import Union, Literal, List, Tuple, Dict
from pathlib import Path

# parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if parent_path not in sys.path:
#     sys.path.append(parent_path) 

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 
from artist.scenario import Scenario
from artist.raytracing import raytracing_utils
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util import utils

from calibration_dataset import CalibrationDataLoader
from logger import TensorboardLogger, TensorboardReader
import util
from util import get_rigid_body_kinematic_parameters_from_scenario, count_parameters, check_for_nan_grad, create_parameter_groups
from image_loss import chamfer_distance_batch_optimized, chamfer_distance_batch, contour_difference

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
# A logger for the calibration model.

random_seed = 7


torch.set_printoptions(precision=16)

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
    def __init__(
            self,
            name: str,
            scenario: Scenario,
            heliostat_ids: List[str],
            calib_data_directory: str,
            config: dict[str, dict],
            # calibration_ids: List[str] = None,
            random_seed: int = random_seed,
            device: Union[torch.device, str] = "cuda"
    ) -> None:
        
        super(CalibrationModel, self).__init__()
        log.info(f"Initializing a Heliostat Calibration Model: {name}")

        model_config = config['model']
        self.name = name
        self.scenario = scenario
        self.heliostat_field = scenario.heliostat_field
        self.target_areas = {area.name: area for area in scenario.target_areas.target_area_list}
        
        self.use_raytracing = bool(config['run']['use_raytracing'])
        self.has_ideal_flux_centers = bool(config['run']['use_ideal_flux_centers'])
        self.random_seed = random_seed
        self.device = device

        # Set up a raytracer. Before that, heliostat needs to be aligned (doesn't matter with which direction).
        self.heliostat_field.align_surfaces_with_incident_ray_direction(
            incident_ray_direction = torch.tensor(
                [0.0, 1.0, 0.0, 0.0], device=device),  # south direction
                device=device
            )
        
        # TODO: Implement distributed environment, paste world_size and rank here.
        self.raytracer = HeliostatRayTracer(
            scenario=self.scenario,
            world_size=1,
            rank=0,
            batch_size=1,
            random_seed=random_seed
            )

        self.calibration_data_loader = CalibrationDataLoader(
                    data_directory=calib_data_directory,
                    heliostats_to_load=heliostat_ids,
                    power_plant_position=self.scenario.power_plant_position,
                    load_flux_images=self.use_raytracing,
                    flux_file_ends_with='-flux.png',
                    properties_file_ends_with='-properties.json',
                    ideal_flux_center=self.has_ideal_flux_centers,
                    device=device
                    )
        self.calibration_data_loader.sun_positions_splits(config=config['splits'])

        self.learnable_parameters = (
            get_rigid_body_kinematic_parameters_from_scenario(
                kinematic=self.heliostat_field.rigid_body_kinematic
            )
        )

        total_parameters, total_elements = count_parameters(self.learnable_parameters)
            
        log.info(f"Total number of parameters and elements: {total_parameters}, {total_elements}")

        self.optimizer, self.scheduler = self.setup_optimizer(optimizer=model_config['optimizer_type'],
                                                              scheduler=model_config['scheduler_type'],
                                                              # param_groups=model_config['param_groups'],  #TODO: complete model config
                                                              config=model_config
                                                              )

        self.save_dir = Path(rf'/dss/dsshome1/05/di38kid/data/results/runs/{self.name}')
        self.tb_logger = TensorboardLogger(name=self.name, log_dir=self.save_dir/'log')
        self.loss_type = None

    def replace_scenario(
        self,
        new_scenario
    ):
        log.info(f"Replacing scenario in {self.name}.")
        self.scenario = new_scenario
        self.heliostat_field = new_scenario.heliostat_field
        self.target_areas = {area.name: area for area in new_scenario.target_areas.target_area_list}
        
        self.heliostat_field.align_surfaces_with_incident_ray_direction(
            incident_ray_direction = torch.tensor(
                [0.0, 1.0, 0.0, 0.0], device=self.device),  # south direction
                device=self.device
            )
        
        # TODO: Implement distributed environment, paste world_size and rank here.
        self.raytracer = HeliostatRayTracer(
            scenario=self.scenario,
            world_size=1,
            rank=0,
            batch_size=1,
            random_seed=self.random_seed
            )
        
        self.learnable_parameters = (
            get_rigid_body_kinematic_parameters_from_scenario(
                kinematic=self.heliostat_field.rigid_body_kinematic
            )
        )

        self.optimizer.params = self.learnable_parameters.parameters()
    
    def setup_optimizer(
            self,
            optimizer: Literal['Adam'] = 'Adam',
            scheduler: Literal['ReduceLROnPlateau', 'CyclicLR'] = 'ReduceLROnPlateau',
            config = {}
    ):  
        initial_lr =config['init_learning_rate']
        max_lr = config['max_lr']
        min_lr = config['min_lr']
        optimizer_kwargs = []
        param_groups = config['parameter_grouping']
        # param_groups = create_parameter_groups(self.learnable_parameters, num_heliostats=self.heliostat_field.number_of_heliostats)
        # TODO: Set-up optimizers and schedulers nicely
        if isinstance(optimizer, str):
            if optimizer == 'Adam':
                if len(param_groups) == 0:
                    self.optimizer = torch.optim.Adam(
                        self.learnable_parameters.parameters(), lr=initial_lr[0]
                    )
                # self.optimizer = torch.optim.Adam(optimizer_kwargs)
            elif optimizer == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                        self.learnable_parameters.parameters(), lr=initial_lr[0]
                    )
            else:
                raise ValueError(
                    f"Optimizer name not found, change name or include new optimizer."
                    )
        else:
            raise ValueError(f"Optimizer must be given as str.")
        
        schedulers = []
        if not scheduler == "None":
            warmup_scheduler = torch.optim.lr_scheduler.ConstantLR(
                        optimizer=self.optimizer,
                        factor=config['warmup_lr_factor'], 
                        total_iters=config['warmup_epochs']
                    )
            schedulers.append(warmup_scheduler)

            if scheduler == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=self.optimizer,
                    factor=config['lr_factor'],
                    patience=config['lr_patience'],
                    factor=0.5,
                )
            elif scheduler == 'CyclicLR':
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer=self.optimizer,
                    base_lr=min_lr[0],
                    max_lr=max_lr[0],
                    step_size_up=config['num_epochs'] - config['warmup_epochs'],
                    mode='triangular',
                    gamma=1.0,
                    scale_mode='cycle',
                    cycle_momentum=False,
                    base_momentum=0.8,
                    max_momentum=0.9,
                )

            elif scheduler == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=150,  # TODO: Use max epochs here
                    eta_min=1e-6                    
                )
            
            elif scheduler == 'CosineAnnealingWarmRestarts ':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts (
                    optimizer=self.optimizer,
                    T_max=150,  # TODO: Use max epochs here
                    eta_min=1e-6                    
                )
                
            else:
                raise ValueError(
                    "Scheduler name not found, change name or implement new scheduler."
                    )
            
            schedulers.append(scheduler)
            
        else:
            constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
                        optimizer=self.optimizer,
                        factor=1, 
                        total_iters=config['num_epochs']
                    )
            schedulers.append(constant_scheduler)

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=self.optimizer,
            schedulers=schedulers,
            milestones=[schedulers[0].total_iters]
        )
        
        log.info(f"Optimizer and Scheduler setup complete.")
        return self.optimizer, self.scheduler

    def forward(
        self,
        heliostats_and_calib_ids: dict[str, List[int]],
        mode: Literal['Train', 'Validation', 'Test', 'None'],  # needed for logging
        epoch: int = 0,
        do_plotting: bool = False,
        device: Union[torch.device, str] = "cuda"
        ):
        """Generate model output for the next step"""
        device = torch.device(device)
        self.tb_logger.set_mode(mode)
        
        field_batch = self.calibration_data_loader.get_field_batch(
            heliostats_and_calib_ids=heliostats_and_calib_ids
        )
        
        # number of samples per heliostat in field
        num_samples = len((field_batch))
        num_heliostats = self.heliostat_field.number_of_heliostats
        
        # Use float64 for all direction and position tensors
        target_reflection_directions = torch.zeros((num_samples, num_heliostats, 4), device=device, dtype=torch.float64)
        target_flux_centers = torch.zeros((num_samples, num_heliostats, 4), device=device, dtype=torch.float64)
        pred_reflection_directions = torch.zeros((num_samples, num_heliostats, 4), device=device, dtype=torch.float64)
        pred_flux_centers = torch.zeros((num_samples, num_heliostats, 4), device=device, dtype=torch.float64)
        
        # Keep bitmaps at original precision (likely float32)
        pred_bitmaps = torch.zeros((num_samples, num_heliostats, 256, 256), device=device)
        diff_bitmaps = torch.zeros_like(pred_bitmaps).to(device=device)
        
        import time
        start = time.time()
        for sample, data in enumerate(field_batch):
            # Convert loaded data to float64
            target_flux_centers[sample] = data['flux_centers'].to(dtype=torch.float64)
            incident_ray_directions = data['incident_rays'].to(dtype=torch.float64)
            target_area_names = data['receiver_targets']
            target_areas = [self.scenario.get_target_area(name) for name in target_area_names]
            
            # Ensure motor positions are float64
            motor_positions_float64 = data['motor_positions'].to(dtype=torch.float64)
            self.heliostat_field.align_surfaces_with_motor_positions(
                motor_positions=motor_positions_float64,
                device=device
            )
            sample_orientations = self.heliostat_field.rigid_body_kinematic.orientations.to(dtype=torch.float64)
            
            for n_heliostat in range(num_heliostats):
                target_area = target_areas[n_heliostat]
                
                if self.has_ideal_flux_centers:  # calibration data contains ideal flux centers
                    target_reflection_directions[sample, n_heliostat] = (
                        data['ideal_flux_centers'][n_heliostat].to(dtype=torch.float64) - sample_orientations[n_heliostat, 0:4, 3]
                        )

                    #print(data['ideal_flux_centers'][n_heliostat])
                    #print(sample_orientations[n_heliostat, 0:4, 3])
                else:
                    target_reflection_directions[sample, n_heliostat] = (
                        data['flux_centers'][n_heliostat].to(dtype=torch.float64) - sample_orientations[n_heliostat, 0:4, 3]
                        )
                    
            if self.use_raytracing:
                # Keep raytracing in original precision for bitmap generation
                sample_bitmaps = self.raytracer.trace_rays_separate(
                    incident_ray_directions=incident_ray_directions,
                    target_areas=target_areas,
                    device=device
                )
                pred_bitmaps[sample] = sample_bitmaps
        
                for n_heliostat in range(num_heliostats):
                    pred_bitmap = sample_bitmaps[n_heliostat]
                    if pred_bitmap.sum() > 0.0001:  # heliostat might miss target area
                        target_area = target_areas[n_heliostat]
                        
                        # Ensure center of mass calculation is in float64
                        center_of_mass = utils.get_center_of_mass(
                                bitmap=sample_bitmaps[n_heliostat],
                                target_center=target_area.center.to(dtype=torch.float64) if hasattr(target_area.center, 'to') else target_area.center,
                                plane_e=target_area.plane_e.to(dtype=torch.float64) if hasattr(target_area.plane_e, 'to') else target_area.plane_e,
                                plane_u=target_area.plane_u.to(dtype=torch.float64) if hasattr(target_area.plane_u, 'to') else target_area.plane_u,
                                device=device
                            ).to(dtype=torch.float64)
                        
                        concentrator_origin = sample_orientations[n_heliostat, 0:4, 3]
                        pred_reflection_direction = (
                            center_of_mass - concentrator_origin
                        )

                    else:
                        # Convert to float64 for the calculations
                        concentrator_normal = sample_orientations[n_heliostat, 0:4, 2]
                        concentrator_normal = concentrator_normal / torch.norm(concentrator_normal, dtype=torch.float64)  # normalize
                        
                        pred_reflection_direction = raytracing_utils.reflect(
                            incident_ray_directions[n_heliostat], concentrator_normal
                            )
                        #print(pred_reflection_direction)
                            
                    pred_reflection_directions[sample][n_heliostat] = pred_reflection_direction
                
            else: # No raytracing, calculate error from orientations.
                concentrator_normals = sample_orientations[:, 0:4, 2]
                # Use float64 for norm calculation
                concentrator_normals = concentrator_normals / torch.norm(concentrator_normals, dim=1, keepdim=True, dtype=torch.float64)  # normalize
                
                for n_heliostat in range(num_heliostats):
                    target_area = target_areas[n_heliostat]
                    # Ensure reflection calculation is in float64
                    reflected_ray = raytracing_utils.reflect(
                        incident_ray_directions[n_heliostat], concentrator_normals[n_heliostat]
                        )
                    #print(reflected_ray)
                    # Ensure intersection calculation is in float64
                    center_point, t = util.calculate_intersection(
                        ray_origin=sample_orientations[n_heliostat, 0:4, 3],
                        ray_direction=reflected_ray,
                        plane_center=target_area.center.to(dtype=torch.float64) if hasattr(target_area.center, 'to') else target_area.center,
                        plane_normal=target_area.normal_vector.to(dtype=torch.float64) if hasattr(target_area.normal_vector, 'to') else target_area.normal_vector,
                    )
                    pred_flux_centers[sample, n_heliostat] = center_point
                    pred_reflection_directions[sample, n_heliostat] = (
                        center_point - sample_orientations[n_heliostat, 0:4, 3]
                        )
                    #print(center_point)
                            
        end = time.time()
        # print('forward loop took:', end-start)
        
        if self.use_raytracing:
        #     # rescale prediction bitmpas
            target_bitmaps = torch.stack([sample['flux_images'] for sample in field_batch]).to(device)
        #     pred_bitmaps = self.rescale_flux_bitmaps(pred_bitmaps, target_bitmaps)
        #     diff_bitmaps = target_bitmaps - pred_bitmaps
        
        with torch.no_grad():  # calculate alignment accurately
            alignment_errors = self.eval_angles_mrad(
                    pred_reflection_directions, 
                    target_reflection_directions
                )
        
        avg_per_heliostat = alignment_errors.mean(dim=0)
        avg_of_field = avg_per_heliostat.mean()
        flat_errors = alignment_errors.flatten()
        median_error = torch.median(flat_errors)
        # print(f"\t[{mode}]: average over whole field: {avg_of_field.item()}")
        # print(f"\tmedian over whole field: {median_error.item()}")
        # print(f"\t[{mode}]: average errors: {avg_per_heliostat.tolist()}")
        self.tb_logger.log_metric("AlignmentErrors_mrad/Average", avg_of_field.item(), epoch)
        self.tb_logger.log_metric("AlignmentErrors_mrad/Median", median_error.item(), epoch)

        if self.use_raytracing:  # Calculate loss
            angels_for_loss = self.robust_angles_mrad(
                                    pred_reflection_directions, 
                                    target_reflection_directions
                                )
            
            mae = torch.nn.L1Loss()
            # mean absolute aligment eror [mrad]
            maae = mae(errors_for_loss, torch.ones_like(angels_for_loss, dtype=torch.float64))
            
            mse = torch.nn.MSELoss()
            # mean squared alignment error [mrad **2]
            msae = mse(errors_for_loss, torch.ones_like(angels_for_loss, dtype=torch.float64)) 
            loss = msae
            
            rmse = torch.sqrt(msae)
            print(f"\tMSAE: {msae.item()}")
    
            # Calculate per-heliostat MAE on bitmaps
            # mspe_per_heliostat = torch.mean(torch.square(pred_bitmaps - target_bitmaps), dim=(0, 2, 3))
            # Get overall MAE from per-heliostat values if needed
            # overall_mspe = torch.mean(mspe_per_heliostat)
            
            mape = diff_bitmaps.abs().mean()
            self.tb_logger.log_loss("MAAE", maae.item(), epoch)
            self.tb_logger.log_loss("MSAE", msae.item(), epoch)
            self.tb_logger.log_loss("RMSAE", rmse.item(), epoch)
            self.tb_logger.log_loss("MeanCHD+MAAE", loss.item(), epoch)

        else:  # Calculate Loss
            l1 = torch.nn.L1Loss()
            mse = torch.nn.MSELoss()
            angles_mrad_for_loss, cos_angles_for_loss = self.robust_angles_mrad(
                                    pred_reflection_directions, 
                                    target_reflection_directions
                                    )
            msae_cos = mse(cos_angles_for_loss, torch.ones_like(cos_angles_for_loss, dtype=torch.float64))
  
            msae = mse(angles_mrad_for_loss, torch.zeros_like(angles_mrad_for_loss, dtype=torch.float64))
            
            msae_center = mse(pred_flux_centers, target_flux_centers)
            
            rmsae = torch.sqrt(msae)

            self.tb_logger.log_loss("MSAE", msae.item(), epoch)
            self.tb_logger.log_loss("MSAE_cos", msae_cos.item(), epoch)
            self.tb_logger.log_loss("MSAE_center", msae_center.item(), epoch)
            self.tb_logger.log_loss("RMSAE", rmsae.item(), epoch)
        
        # Log metrics
        for n_heliostat, (heliostat_id, calib_ids) in enumerate(heliostats_and_calib_ids.items()):
            self.tb_logger.log_heliostat_metric("AlignmentErrors_mrad", 
                                            heliostat_id, 
                                            avg_per_heliostat.tolist()[n_heliostat], 
                                            epoch)
            
            for n_sample, (calib_id, error) in enumerate(zip(calib_ids, alignment_errors[:, n_heliostat].tolist())):
                self.tb_logger.log_heliostat_metric("AlignmentErrors_mrad", 
                                                heliostat_id, 
                                                error, 
                                                epoch, 
                                                calib_id=calib_id)
                    
        return alignment_errors, msae_cos

    @staticmethod
    def robust_angles_mrad(
            v1: torch.Tensor,
            v2: torch.Tensor,
            epsilon: float = 1e-10
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        # in case v1 or v2 have shape (4,), bring to (1, 4)
        assert v1.shape == v2.shape, 'Given Tensors must have identical shapes.'
        
        # Ensure v1 and v2 are float64
        v1 = v1.to(dtype=torch.float64)
        v2 = v2.to(dtype=torch.float64)
        
        while v1.dim() < 3:  
            v1 = v1.unsqueeze(0)  # bring to [B, H, 4] 
            v2 = v2.unsqueeze(0) 

        # Use float64 for all vector operations
        m1 = torch.norm(v1, dim=-1, dtype=torch.float64)
        m2 = torch.norm(v2, dim=-1, dtype=torch.float64)
        dot_products = torch.sum(v1 * v2, dim=-1)
        cos_angles = dot_products / (m1 * m2 + epsilon)  # avoid division by zero
        angles_rad = torch.acos(
            torch.clip(cos_angles, -1.0 + 1e-7, 1.0 - 1e-7)
        )
        return (angles_rad * 1000), cos_angles

    @staticmethod
    def eval_angles_mrad(
            v1: torch.Tensor,
            v2: torch.Tensor,
            ) -> torch.Tensor:
        # Ensure v1 and v2 are float64
        v1 = v1.to(dtype=torch.float64)
        v2 = v2.to(dtype=torch.float64)
        
        # Use float64 for all vector operations
        m1 = torch.norm(v1, dim=-1, dtype=torch.float64)
        m2 = torch.norm(v2, dim=-1, dtype=torch.float64)
        dot_products = torch.sum(v1 * v2, dim=-1)
        cos_angles = dot_products / (m1 * m2)
        angles_rad = torch.acos(
            torch.clip(cos_angles, -1.0, 1.0)
        )
        return angles_rad * 1000  # Convert to milliradians
    
    def freeze_parameters(
        self,
        freeze_parameters: List[str] = []
    ):
        # Iterate over the nested parameters and freeze / unfreeze
        # Names and parameter lists in the learnable parameters dict
        for name, param_group in self.learnable_parameters.items():
            grad = True  
            # Check if the parameter group 
            if name in freeze_parameters:
                log.info(f"Freezing parameters '{name}'.") 
                grad = False
            for params in param_group:
                if isinstance(params, (torch.nn.ParameterList, list, tuple)):
                    for param in params:
                        param.requires_grad = grad
                        has_grad = param.requires_grad    
                else:
                    params.requires_grad = grad
                    has_grad = params.requires_grad    
    
    def phase_learning(
        self,
        epoch: int,
        phase_parameter_learning: Dict[str, List]
    ):
        parameter_names = list(self.learnable_parameters.keys())
        phased_param_groups = list(phase_parameter_learning.values())
        
        # create new list which will contain the parameters for freezing in each phase
        phased_freeze_parameters = []
        for groups in phased_param_groups:
            # look up the parameters which have matching names to the given groups in each phase
            freeze_parameters = [param for param in parameter_names if not any(group in param for group in groups)]
            phased_freeze_parameters.append(freeze_parameters)
        
        # phased freezing in warm up
        if epoch == 1:
            self.freeze_parameters(phased_freeze_parameters[0])
        elif epoch == 200:
            self.freeze_parameters(phased_freeze_parameters[1])
        elif epoch == 400:
            self.freeze_parameters(phased_freeze_parameters[2])
        elif epoch == 600:
            self.freeze_parameters([])  # unfreeze for last 400 epochs of warm up

        # phased freezing for cyclic LR
        elif epoch >= 1000:
            self.freeze_parameters([])
            # phased learning of 50 epochs per parameter group
            if epoch in [1000 + i * 150 for i in range(4)]:
                self.freeze_parameters(phased_freeze_parameters[0])
            elif epoch in [1050 + i * 150 for i in range(4)]:
                self.freeze_parameters(phased_freeze_parameters[1])
            elif epoch in [1100 + i * 150 for i in range(4)]:
                self.freeze_parameters(phased_freeze_parameters[2])
                
            elif epoch >= 1600:
                self.freeze_parameters([])  # unfreeze all for the last 1250 epochs
    
    def calibrate(
            self,
            split_type: str,
            split_sizes: Tuple[int, int],
            heliostat_names: list = [],
            num_epochs: int = 30,
            log_steps: int = 10,
            tolerance: float = 0.0,
            seed: int = random_seed,
            freeze_parameters: List[str] = (),
            phase_parameter_learning: Dict[str, List] = {},
            run_config: dict = None,
            device: Union[torch.device, str] = "cuda"
    ):

        split_df = self.calibration_data_loader.splits[split_type][split_sizes]
        
        self.freeze_parameters(freeze_parameters)

        log.info(f"Starting Calibration with motor positions...")
        epoch = 1
        avg_valid_error = torch.inf
        while epoch <= num_epochs and avg_valid_error > tolerance:
            
            if len(phase_parameter_learning) > 0:
                self.phase_learning(epoch, phase_parameter_learning)

            self.train()
            self.optimizer.zero_grad()

            train_df = split_df[split_df['Split'] == 'train']
            train_dict = {
                heliostat_id: train_df.loc[train_df['HeliostatId'] == heliostat_id].index.tolist()
                for heliostat_id in heliostat_names
            }
            
            train_alignment_errors, train_loss = self.forward(
                mode='Train', 
                epoch=epoch,
                heliostats_and_calib_ids=train_dict,
                do_plotting=False,
                device=device
            )
            train_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.learnable_parameters.parameters(), max_norm=0.5)            
            
            check_for_nan_grad(self.learnable_parameters)
            self._log_parameters(epoch=epoch)
            self.optimizer.step()

            self.eval()
            with torch.no_grad():
                val_df = split_df[split_df['Split'] == 'validation']
                val_dict = {
                    heliostat_id: val_df.loc[val_df['HeliostatId'] == heliostat_id].index.tolist()
                    for heliostat_id in heliostat_names
                }
            
                
                valid_alignment_errors, valid_loss = self.forward(
                    mode='Validation', epoch=epoch,
                    heliostats_and_calib_ids=val_dict,
                    do_plotting=False,
                    device=device
                )
            self.scheduler.step()

            avg_train_error = train_alignment_errors.mean()
            avg_valid_error = valid_alignment_errors.mean()

            if epoch % log_steps == 0:
                log.info(f"Epoch: {epoch} of {num_epochs} max, "
                         f"Train / Valid Loss: {train_loss.item()} / {valid_loss.item()}, "
                         f"Avg Train / Valid Error (mrad): {avg_train_error.item():.4f} / {avg_valid_error.item():.4f}, "
                         f"LR: {self.scheduler.get_last_lr()}"
                         )

            epoch += 1
        
        self._log_parameters(epoch=epoch)
        
        test_df = split_df[split_df['Split'] == 'test']
        test_dict = {
            heliostat_id: test_df.loc[test_df['HeliostatId'] == heliostat_id].index.tolist()
            for heliostat_id in heliostat_names
        }
        self.evaluate_model_fit(test_dict, num_epochs, device=device)

    def evaluate_model_fit(
            self,
            evaluate_dict: dict[str, list[int]],
            epoch: int,
            do_plotting : bool = False,
            device: Union[torch.device, str] = "cuda"
    ):
        self.eval()
        with torch.no_grad():
            test_alignment_errors, test_loss = self.forward(
                    mode='Test', epoch=epoch,
                    heliostats_and_calib_ids=evaluate_dict,
                    do_plotting=do_plotting,
                    device=device
                )

            avg_test_error = test_alignment_errors.mean()

            log.info(f"Test Loss: {test_loss.item()}, "
                     f"Avg Test Error (mrad): {avg_test_error.item():.4f}")

    def _log_parameters(self, epoch, obj=None, name=None, heliostat_idx=None, index=None):
        if obj is None:
            # Log all parameters as scalars
            obj = self.learnable_parameters
        else:
            # Handle the previous recursive logic for backward compatibility
            if isinstance(obj, torch.nn.Parameter):
                heliostat_name = self.heliostat_field.all_heliostat_names[heliostat_idx]
                self.tb_logger.log_parameter_as_scalar(name, heliostat_name, obj, epoch, index)
            
        if isinstance(obj, (torch.nn.ParameterList, list, tuple)):
            if heliostat_idx is None:
                if len(obj) == len(self.heliostat_field.all_heliostat_names):
                    for h, param in enumerate(obj):
                        self._log_parameters(epoch=epoch, obj=param, name=name, heliostat_idx=h)
                else:
                    for i, param in enumerate(obj):
                        self._log_parameters(epoch=epoch, obj=param, name=name, heliostat_idx=h, index=i)
                
            else: 
                [self._log_parameters(epoch=epoch, obj=param, name=name, heliostat_idx=heliostat_idx, index=i) 
                 for i, param in enumerate(obj)]

        if isinstance(obj, (torch.nn.ParameterDict, dict)):
            for name, obj in obj.items():
                self._log_parameters(epoch=epoch, obj=obj, name=name)
    
    def analyze_calibration_results(model_dir=None):
        """
        Analyze the results of a calibration run.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        log_dir : str, optional
            Directory where logs are saved
        """
        if model_dir is None:
            model_dir = self.model_dir
        
        # Create a TensorboardReader
        reader = TensorboardReader(f'{model_dir}/log')
        
        # Get a summary of the final performance
        summary = reader.create_performance_summary()
        print("Performance Summary:")
        print(summary)
        
        plot_dir = f'{model_dir}/plots'
        os.makedirs(plot_dir, exist_ok=True)
        # Plot the loss curves
        loss_fig = reader.plot_losses(modes=['Train', 'Validation'])
        loss_fig.savefig(f'{plot_dir}/01_loss_over_epochs.png')
        
        # Plot alignment errors
        error_fig = reader.plot_metrics(metric_name='AlignmentErrors_mrad', modes=['Train', 'Validation'])
        loss_fig.savefig(f'{plot_dir}/02_error_over_epochs.png')
        
        # Plot final error distribution
        # TODO: Here create plot for sun distribution
        dist_fig = reader.visualize_final_error_distribution()
        dist_fig.savefig(f'/dss/dsshome1/05/di38kid/data/results/plots/{model_name}_error_distribution.png')

    @staticmethod
    def rescale_flux_bitmaps(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
    
        assert predictions.shape == targets.shape, "Predicted bitmaps must match the shape of the target bitmpas."

        pred_energy = torch.sum(predictions, dim=(-2, -1), keepdim=True)  # Shape: [B, H, 1, 1]
        target_energy = torch.sum(targets, dim=(-2, -1), keepdim=True)  # Shape: [B, H, 1, 1]  
        
        scaling_factors = torch.ones_like(pred_energy)  # Default scaling factor is 1.0
        non_zero_mask = pred_energy > 1e-4  # Find non-zero prediction energies
        
        # Apply scaling only to non-zero predictions
        scaling_factors[non_zero_mask] = target_energy[non_zero_mask] / pred_energy[non_zero_mask]
        
        return predictions * scaling_factors

    
    def find_max_lr(
        self,
        heliostat_and_calib_dict: dict(),
        restore_scenario,
        start_lr: float = 1e-2,
        end_lr: float = 100,
        num_iterations = 100,
        freeze_parameters = []
    ):
        
        # Freeze / unfreeze parameters depending on selection
        self.freeze_parameters(freeze_parameters)
            
        # Use Adam optimizer for finding max learning rate with 
        optimizer = torch.optim.Adam(
                self.learnable_parameters.parameters(), lr=start_lr
                )
            
        # Set starting learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = start_lr
            
        # Determine the exponential factor for increasing the learning rate
        gamma = (end_lr / start_lr) ** (1 / num_iterations)
        
        lrs = []
        losses = []
        
        self.train()
        
        log.info("Searching for maximum learning rate...")
        for i in range(num_iterations):

            
            optimizer.zero_grad()
                
            _, loss = self.forward(
                mode='None',  # disable logging
                heliostats_and_calib_ids=heliostat_and_calib_dict,
                device=self.device
            )
            if loss.isnan():
                log.info(f"Loss was nan at iteration {i} - search was stopped.")
                break  # break search loop
            loss.backward()
            optimizer.step()
            
            lr = optimizer.param_groups[0]['lr']
            lrs.append(lr)
            losses.append(loss.item())
            
            # Increase learning rate for next iteration
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * gamma
                
            if i % 10 == 0:
                log.info(f"Iteration {i} / {num_iterations}, LR: {lr:.8f}, Loss: {loss.item():.8f}")
        
        # Store deep copy of kinematic class with its initial parameters
        self.replace_scenario(restore_scenario)
        
        from matplotlib import pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, losses)
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss: MSE (1 - cos(alignment error))')
        plt.title('Learning Rate Range Test')
        plt.grid(True)
        # Create output directory if it doesn't exist
        plots_dir = self.save_dir / 'plots'
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(plots_dir / 'learning_rate_range_test.png')    
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.loglog(lrs, losses)
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss: MSE [1 - cos(alignment error)]')
        plt.title('Learning Rate Range Test')
        plt.grid(True)
        # Create output directory if it doesn't exist
        plots_dir = self.save_dir / 'plots'
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(plots_dir / 'learning_rate_range_test__log.png')    
        plt.close()
        
        log.warn("Restart calibration after using find_max_lr!")
            
            
    
    
    # TODO: Include Learning Rate finder.
    # def find_lr(
    #         self,
    #         start_lr: float = 1e-8,
    #         end_lr: float = 0.1,
    #         num_iter: int = 1,
    #         device: Union[torch.device, str] = "cuda"
    # ) -> None:
    #     device = torch.device(device)
    #     lrs = torch.logspace(torch.log10(torch.tensor(start_lr)),
    #                          torch.log10(torch.tensor(end_lr)), steps=num_iter).tolist()
    #
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer,
    #                                                   lr_lambda=lambda x: lrs[min(x, num_iter - 1)])
    #     losses, log_lrs = [], []
    #
    #     log.info("Finding Learning Rate...")
    #     for i in range(len(lrs)):
    #         total_loss = 0
    #         for calibration in self.calibration_data:
    #             self._optimizer.zero_grad()
    #             preferred_reflection_direction_calibration = (
    #                     calibration[1]['center_image'] - self.heliostat.position
    #             )
    #             preferred_reflection_direction_calibration = (
    #                     preferred_reflection_direction_calibration / torch.norm(preferred_reflection_direction_calibration)
    #             )
    #             preferred_reflection_direction = self.forward(calibration, device)
    #             loss = self.calculate_loss_mrad(
    #                 preferred_reflection_direction,
    #                 preferred_reflection_direction_calibration)
    #             loss.backward()
    #             self._optimizer.step()
    #             total_loss += loss.item()
    #         losses.append(total_loss / len(self.calibration_data))
    #         log_lrs.append(torch.log10(torch.tensor(lrs[i])))
    #         scheduler.step()
    #
    #     best_lr = lrs[losses.index(min(losses))]
    #     print(f"Best learning rate found: {best_lr}")
    #     self._optimizer.param_groups[0]["lr"] = best_lr
    #
    #     plt.plot(lrs, losses)
    #     plt.xlabel("Log Learning Rate")
    #     plt.ylabel("Loss")
    #     plt.title("Learning Rate Finder")
    #     plt.show()


