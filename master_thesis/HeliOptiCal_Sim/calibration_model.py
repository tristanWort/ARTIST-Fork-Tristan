import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity 
from torch.utils.tensorboard import SummaryWriter 
# import tensorflow.summary as summary
import logging
import sys
import os

from torchmetrics.image import StructuralSimilarityIndexMeasure
from typing import Union, Literal, List, Tuple
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
                    flux_file_ends_with='-simulated-flux.png',
                    properties_file_ends_with='-simulation-properties.json',
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
                                                              initial_lr=model_config['init_learning_rate'],
                                                              )

        self.log_dir=Path(rf'/dss/dsshome1/05/di38kid/data/results/runs/{self.name}')
        self.tb_logger = TensorboardLogger(name=self.name, log_dir=self.log_dir/'log')
        self.loss_type = None

    def setup_optimizer(
            self,
            optimizer: Literal['Adam'] = 'Adam',
            scheduler: Literal['ReduceLROnPlateau', 'CyclicLR'] = 'ReduceLROnPlateau',
            param_groups: List[List[str]] = None,
            initial_lr: Union[float, List] = 0.00001,
            lr_factor: float = 0.1,
            lr_patience: int = 10,
            lr_threshold: float = 0.1,
            threshold_mode: Literal['abs', 'rel'] = 'abs'
    ):
        optimizer_kwargs = []
        scheduler_kwargs = {'base_lr': None, 'max_lr': None}
        # param_groups = create_parameter_groups(self.learnable_parameters, num_heliostats=self.heliostat_field.number_of_heliostats)
        # TODO: Set-up optimizers and schedulers nicely
        if isinstance(optimizer, str):
            if optimizer == 'Adam':
                if param_groups is None:
                    self.optimizer = torch.optim.Adam(
                        self.learnable_parameters.parameters(), lr=initial_lr[0]
                    )
                else:
                    for i, param_group in enumerate(param_groups):
                        params = [self.learnable_parameters[param] for param in param_group]
                        optimizer_kwargs.append({'params': params, 'lr': initial_lr[i]})
                    scheduler_kwargs['base_lr'] = [lr / 10 for lr in initial_lr]
                    scheduler_kwargs['max_lr'] = [lr for lr in initial_lr]
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
        if scheduler is not None:
            if scheduler == 'ReduceLROnPlateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=self.optimizer,
                    factor=lr_factor,
                    patience=lr_patience,
                    threshold=lr_threshold,
                    threshold_mode=threshold_mode
                )
            elif scheduler == 'CyclicLR':
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer=self.optimizer,
                    base_lr=scheduler_kwargs['base_lr'],
                    max_lr=scheduler_kwargs['max_lr'],
                    step_size_up=100,
                    step_size_down=50,
                    mode='triangular2',
                    gamma=1.0,
                    scale_mode='cycle',
                    cycle_momentum=False,
                    base_momentum=0.8,
                    max_momentum=0.9,
                    last_epoch=-1
                )
            elif scheduler == 'OneCycleLR':
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=self.optimizer,
                    total_steps=150,
                    max_lr=0.01,
                    div_factor=1e-6 / 1e-10
                )
            elif scheduler == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=150,  # TODO: Use max epochs here
                    eta_min=1e-6                    
                )
                
            else:
                raise ValueError(
                    "Scheduler name not found, change name or implement new scheduler."
                    )
        log.info(f"Optimizer and Scheduler setup complete.")
        return self.optimizer, self.scheduler

    def forward(
            self,
            heliostats_and_calib_ids: dict[str, List[int]],
            mode: Literal['Train', 'Validation', 'Test'],  # needed for logging
            epoch: int,
            do_plotting: bool = False,
            device: Union[torch.device, str] = "cuda"
            ):
        """Generate model output for the next step"""
        device = torch.device(device)
        # self.heliostat_field.is_aligned = True
        
        field_batch = self.calibration_data_loader.get_field_batch(
            heliostats_and_calib_ids=heliostats_and_calib_ids
        )
        
        # number of samples per heliostat in field
        num_samples = len((field_batch))
        num_heliostats = self.heliostat_field.number_of_heliostats
        target_reflection_directions = torch.zeros((num_samples, num_heliostats, 4), device=device)
        target_flux_centers = torch.zeros((num_samples, num_heliostats, 4), device=device)
        pred_reflection_directions = torch.zeros((num_samples, num_heliostats, 4), device=device)
        pred_flux_centers = torch.zeros((num_samples, num_heliostats, 4), device=device)
        pred_bitmaps = torch.zeros((num_samples, num_heliostats, 256, 256), device=device)
        diff_bitmaps = torch.zeros_like(pred_bitmaps).to(device=device)
        
        import time
        start = time.time()
        for sample, data in enumerate(field_batch):
            target_flux_centers[sample] = data['flux_centers']
            incident_ray_directions = data['incident_rays']
            target_area_names = data['receiver_targets']
            target_areas = [self.scenario.get_target_area(name) for name in target_area_names]
            
            self.heliostat_field.align_surfaces_with_motor_positions(
                motor_positions=data['motor_positions'],
                device=device
            )
            sample_orientations = self.heliostat_field.rigid_body_kinematic.orientations
            
            for n_heliostat in range(num_heliostats):
                target_area = target_areas[n_heliostat]
                
                if self.has_ideal_flux_centers:  # calibration data contains ideal flux centers
                    target_reflection_directions[sample, n_heliostat] = (
                        data['ideal_flux_centers'][n_heliostat] - sample_orientations[n_heliostat, 0:4, 3]
                        )
                else:
                    target_reflection_directions[sample, n_heliostat] = (
                        data['flux_centers'][n_heliostat] - sample_orientations[n_heliostat, 0:4, 3]
                        )
                    
            if self.use_raytracing:
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
                        
                        center_of_mass = utils.get_center_of_mass(
                                bitmap=sample_bitmaps[n_heliostat],
                                target_center=target_area.center,
                                plane_e=target_area.plane_e,
                                plane_u=target_area.plane_u,
                                device=device
                            )
                        
                        concentrator_origin = sample_orientations[n_heliostat, 0:4, 3]
                        pred_reflection_direction = (
                            center_of_mass - concentrator_origin
                        )

                    else:
                        # TODO: Remove print after debugging
                        # print(f"Energy in flux pred low for: {self.heliostat_field.all_heliostat_names[n_heliostat]}:  {sample}")
                        concentrator_normal = sample_orientations[n_heliostat, 0:4, 2]
                        concentrator_normal = concentrator_normal / torch.norm(concentrator_normal)  # normalize
                        
                        pred_reflection_direction = raytracing_utils.reflect(
                            incident_ray_directions[n_heliostat], concentrator_normal
                            )
    
                        # TODO: Remove empty bitmap and use simple alignment loss
                    pred_reflection_directions[sample][n_heliostat] = pred_reflection_direction
                
            else: # No raytracing, calculate error from orientations.
                concentrator_normals = sample_orientations[:, 0:4, 2]
                concentrator_normals = concentrator_normals / torch.norm(concentrator_normals, dim=1, keepdim=True)  # normalize
                for n_heliostat in range(num_heliostats):
                    # pred_reflection_directions[sample, n_heliostat] = raytracing_utils.reflect(
                    #     incident_ray_directions[n_heliostat], concentrator_normals[n_heliostat]
                    #     )
                    target_area = target_areas[n_heliostat]
                    reflected_ray = raytracing_utils.reflect(
                        incident_ray_directions[n_heliostat], concentrator_normals[n_heliostat]
                        )
                    center_point, t = util.calculate_intersection(
                        ray_origin=sample_orientations[n_heliostat, 0:4, 3],
                        ray_direction=reflected_ray,
                        plane_center=target_area.center,
                        plane_normal=target_area.normal_vector,
                    )
                    pred_flux_centers[sample, n_heliostat] = center_point
                    pred_reflection_directions[sample, n_heliostat] = (
                        center_point - sample_orientations[n_heliostat, 0:4, 3]
                        )
                             
        end = time.time()
        # print('forward loop took:', end-start)
        
        if self.use_raytracing:
        #     # rescale prediction bitmpas
            target_bitmaps = torch.stack([sample['flux_images'] for sample in field_batch]).to(device)
        #     pred_bitmaps = self.rescale_flux_bitmaps(pred_bitmaps, target_bitmaps)
        #     diff_bitmaps = target_bitmaps - pred_bitmaps
            
        #     contour_bitmaps = contour_difference(predictions=pred_bitmaps, 
        #                                         targets=target_bitmaps, 
        #                                         threshold=0.5, 
        #                                         sharpness=20.0)
        
        if do_plotting:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(nrows=num_heliostats,
                                    ncols=num_samples, 
                                    figsize=(150, 10))
            for s in range(num_samples):
                for h in range(num_heliostats):
                    heliostat_name = self.heliostat_field.all_heliostat_names[h]
                    cal_id = heliostats_and_calib_ids[heliostat_name][s]
                    axs[h][s].imshow(pred_bitmaps[s][h].cpu().detach(), cmap="inferno")
                    axs[h][s].set_title(f"Sample {s}, {int(cal_id)}", fontsize=10)
                    axs[h][s].set_xticks([])
                    axs[h][s].set_yticks([])
                    if s == 0:
                        # Add heliostat name as y-label for the first column
                        axs[h][s].set_ylabel(heliostat_name, fontsize=12, rotation=0, 
                                            horizontalalignment='right', verticalalignment='center')
            fig.suptitle(f"Flux Predictions: {self.name}", fontsize=16)
            plt.tight_layout()
            
            plt.subplots_adjust(top=0.9, left=0.1)  # This creates space for the title
            plt.savefig(f"/dss/dsshome1/05/di38kid/data/results/plots/flux_pred.png")

            for s in range(num_samples):
                for h in range(num_heliostats):
                    axs[h][s].imshow(target_bitmaps[s][h].cpu().detach(), cmap="inferno")
            fig.suptitle(f"Ground Truth: {self.name}", fontsize=16)
            plt.tight_layout()
            
            plt.subplots_adjust(top=0.9, left=0.1)  # This creates space for the title
            plt.savefig(f"//dss/dsshome1/05/di38kid/data/results/plots/flux_target.png") 

            for s in range(num_samples):
                for h in range(num_heliostats):
                    axs[h][s].imshow(diff_bitmaps[s][h].cpu().detach(), cmap="inferno")
            fig.suptitle(f"Diff Flux (Prediction - Ground Truth): {self.name}", fontsize=16)
            plt.tight_layout()
            
            plt.subplots_adjust(top=0.9, left=0.1)  # This creates space for the title
            plt.savefig(f"//dss/dsshome1/05/di38kid/data/results/plots/flux_diff.png")        
        
        with torch.no_grad():  # calculate alignment accurately
            alignment_errors = self.eval_angles_mrad(
                    pred_reflection_directions, 
                    target_reflection_directions
                )
            
        # print(f"\talignment errors:")
        # for sample, field_errors in enumerate(alignment_errors.tolist()):
        #     print(f'\t\t[{sample}]: {field_errors}')
        # print()
        
        avg_per_heliostat = alignment_errors.mean(dim=0)
        print(f"\t[{mode}]: average errors: {avg_per_heliostat.tolist()}")
        
        avg_of_field = avg_per_heliostat.mean()
        print(f"\t[{mode}]: average over whole field: {avg_of_field.item()}")
        self.tb_logger.log_metric("AlignmentErrors_mrad/Average", avg_of_field.item(), mode, epoch)

        flat_errors = alignment_errors.flatten()
        median_error = torch.median(flat_errors)
        self.tb_logger.log_metric("AlignmentErrors_mrad/Median", median_error.item(), mode, epoch)
        # print(f"\tmedian over whole field: {median_error.item()}")

        if self.use_raytracing:
            angels_for_loss = self.robust_angles_mrad(
                                    pred_reflection_directions, 
                                    target_reflection_directions
                                )
            
            mae = torch.nn.L1Loss()
            # mean absolute aligment eror [mrad]
            maae = mae(errors_for_loss, torch.ones_like(angels_for_loss))
            
            mse = torch.nn.MSELoss()
            # mean squared alignment error [mrad **2]
            msae = mse(errors_for_loss, torch.ones_like(angels_for_loss)) 
            loss = msae
            
            rmse = torch.sqrt(msae)
            print(f"\tMSAE: {msae.item()}")
            
            # Calculate per-heliostat MAE on bitmaps
            # mspe_per_heliostat = torch.mean(torch.square(pred_bitmaps - target_bitmaps), dim=(0, 2, 3))
            # Get overall MAE from per-heliostat values if needed
            # overall_mspe = torch.mean(mspe_per_heliostat)
            
            mape = diff_bitmaps.abs().mean()
            # mspe = mse(pred_bitmaps, target_bitmaps)
            # print(f"\tMSPE: {mspe_per_heliostat.tolist()}")
            # print(f"\tField MSPE: {overall_mspe.item()}")
            # print(f"\tField MSPE: {mspe.item()}")
            
            # chamfer_loss = chamfer_distance(flux_bitmap_pred[0], flux_bitmaps[0])
            # ssim_fnc = StructuralSimilarityIndexMeasure().to(device)
            # ssim = ssim_fnc(pred_bitmaps.reshape(-1, 256, 256).unsqueeze(1), 
            #                 target_bitmaps.reshape(-1, 256, 256).unsqueeze(1))
            # print(f"\tSSIM: {ssim.item()}")
            
            # chamfer_distances = chamfer_distance_batch_optimized(pred_bitmaps.reshape(-1, 256, 256), 
            #                                                      target_bitmaps.reshape(-1, 256, 256))
            # non_zero_mask = chamfer_distances != 0.0
            # non_zero_count = non_zero_mask.sum()
            # Calculate mean of non-zero values
            # mean_chd = chamfer_distances[non_zero_mask].sum() / non_zero_count
            # mean_chd = chamfer_distances.mean()
            
            # loss_type = '(1 - Structural Similarity Index)'
            # chamfer_distances = chamfer_distance_batch(pred_bitmaps.reshape(-1, 256, 256), 
            #                                            target_bitmaps.reshape(-1, 256, 256))
            # chd = chamfer_distances.mean()
            
            # print(f"\tCHD: {mean_chd.item()}")
            
            # Calculate combined loss from alignment MSE and flux MSE.
            # alpha = 0.8
            # loss = alpha * msae + (1 - alpha) * mspe
            # TODO: Make alpha learnable https://github.com/Helmholtz-AI-Energy/propulate/blob/main/tutorials/nm_example.py
            # loss = alpha * mean_chd + (1-alpha) * maae 
            
            # self.tb_logger.log_loss("MeanCHD", mean_chd.item(), mode, epoch)
            self.tb_logger.log_loss("MAAE", maae.item(), mode, epoch)
            self.tb_logger.log_loss("MSAE", msae.item(), mode, epoch)
            self.tb_logger.log_loss("RMSAE", rmse.item(), mode, epoch)
            # self.tb_logger.log_loss("MeanSSIM", ssim.item(), mode, epoch)
            # self.tb_logger.log_loss("MAPE", mape.item(), mode, epoch)
            # self.tb_logger.log_loss("MSPE", mspe.item(), mode, epoch)
            self.tb_logger.log_loss("MeanCHD+MAAE", loss.item(), mode, epoch)

        else:
            l1 = torch.nn.L1Loss()
            mse = torch.nn.MSELoss()
            cos_angles_for_loss = self.robust_angles(
                                    pred_reflection_directions, 
                                    target_reflection_directions
                                )
            msae = mse(cos_angles_for_loss, torch.ones_like(cos_angles_for_loss)) * 1000000
            center_loss = l1(pred_flux_centers, target_flux_centers)
            loss = msae
            rmsae = torch.sqrt(msae)
            self.tb_logger.log_loss("MSAE", msae.item(), mode, epoch)
            self.tb_logger.log_loss("RMSAE", rmsae.item(), mode, epoch)
        
        for n_heliostat, (heliostat_id, calib_ids) in enumerate(heliostats_and_calib_ids.items()):
            self.tb_logger.log_heliostat_metric("AlignmentErrors_mrad", 
                                             heliostat_id, 
                                             avg_per_heliostat.tolist()[n_heliostat], 
                                             mode, 
                                             epoch)
            
            for n_sample, (calib_id, error) in enumerate(zip(calib_ids, alignment_errors[:, n_heliostat].tolist())):
                self.tb_logger.log_heliostat_metric("AlignmentErrors_mrad", 
                                                 heliostat_id, 
                                                 error, 
                                                 mode, 
                                                 epoch, 
                                                 calib_id=calib_id)
                if self.use_raytracing:
                    self.tb_logger.log_image("FluxPredictions", 
                                          heliostat_id, 
                                          pred_bitmaps[n_sample, n_heliostat, :, :], 
                                          mode, 
                                          epoch, 
                                          calib_id=calib_id)
                    # if epoch == 1:
                    #     self.tb_logger.log_image("FluxGroundtruths", 
                    #                           heliostat_id, 
                    #                           target_bitmaps[n_sample, n_heliostat, :, :], 
                    
                    #                           mode, 
                    #                           epoch, 
                    #                           calib_id=calib_id)
                    
                    # self.tb_logger.log_image("FluxDiffs", 
                    #                       heliostat_id, 
                    #                       diff_bitmaps[n_sample, n_heliostat, :, :], 
                    #                       mode, 
                    #                       epoch, 
                    #                       calib_id=calib_id)
                    
                    # self.tb_logger.log_image("ContourDiffs", 
                    #                      heliostat_id, 
                    #                      contour_bitmaps[n_sample, n_heliostat, :, :], 
                    #                      mode, 
                    #                      epoch, 
                    #                      calib_id=calib_id)
                    
        return alignment_errors, loss

    @staticmethod
    def robust_angles(
            v1: torch.Tensor,
            v2: torch.Tensor,
            epsilon: float = 1e-10
            ) -> torch.Tensor:
        # in case v1 or v2 have shape (4,), bring to (1, 4)
        assert v1.shape == v2.shape, 'Given Tensors must have identical shapes.'
        while v1.dim() < 3:  
            v1 = v1.unsqueeze(0)  # bring to [B, H, 4] 
            v2 = v2.unsqueeze(0) 

        m1 = torch.norm(v1, dim=-1)
        m2 = torch.norm(v2, dim=-1)
        dot_products = torch.sum(v1 * v2, dim=-1)
        cos_angles = dot_products / (m1 * m2 + epsilon)  # avoid division by zero
        return cos_angles
    
    @staticmethod
    def eval_angles_mrad(
            v1: torch.Tensor,
            v2: torch.Tensor,
            ) -> torch.Tensor:
        m1 = torch.norm(v1, dim=-1)
        m2 = torch.norm(v2, dim=-1)
        dot_products = torch.sum(v1 * v2, dim=-1)
        cos_angles = dot_products / (m1 * m2)
        angles_rad = torch.acos(
            torch.clip(cos_angles, -1.0, 1.0)
        )
        return angles_rad * 1000

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
            run_config: dict = None,
            device: Union[torch.device, str] = "cuda"
    ):

        split_df = self.calibration_data_loader.splits[split_type][split_sizes]
        
        for name, param_group in self.learnable_parameters.items():
            if name in freeze_parameters:
                for params in param_group:
                    if isinstance(params, (torch.nn.ParameterList, list, tuple)):
                        for param in params:
                            if isinstance(param, (torch.nn.ParameterList, list, tuple)):
                                for p in param:
                                    p.requires_grad = False
                            else:
                                param.requires_grad = False
                                
                    else:
                        params.requires_grad = False
                print(f"Parameters '{name}' have been frozen")

        log.info(f"Starting Calibration with motor positions...")
        epoch = 1
        avg_valid_error = torch.inf
        while epoch <= num_epochs and avg_valid_error > tolerance:
            
            if epoch > 90:
                self.heliostat_field.rigid_body_kinematic.all_deviations_params['first_joint_tilt_n'][0].data = torch.tensor(0.0, device=self.device)
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
            self.scheduler.step(valid_loss)

            avg_train_error = train_alignment_errors.mean()
            avg_valid_error = valid_alignment_errors.mean()

            if epoch % log_steps == 0:
                log.info(f"Epoch: {epoch} of {num_epochs} max, "
                         f"Train / Valid Loss: {train_loss.item():.4f} / {valid_loss.item():.4f}, "
                         f"Avg Train / Valid Error (mrad): {avg_train_error.item():.2f} / {avg_valid_error.item():.2f}, "
                         # f"LR: {self.scheduler.get_last_lr()}"
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

            # summary.scalar("Loss/Test", test_loss.detach().numpy(), epoch)
            # summary.scalar("AvgError/Test", avg_test_error.detach().numpy(), epoch)

            # for calib_id, error in zip(calibration_ids, test_alignment_errors):
            #     summary.scalar(f"Errors/Valid/{calib_id}", error.detach().numpy(), epoch)

            log.info(f"Test Loss: {test_loss.item():.4f}, "
                     f"Avg Test Error (mrad): {avg_test_error.item():.2f}")

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


