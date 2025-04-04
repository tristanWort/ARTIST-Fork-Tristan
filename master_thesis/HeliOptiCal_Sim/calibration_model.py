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
                    load_flux_images=False,
                    has_surface_normal=False,
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

        self.writer = SummaryWriter(log_dir=f'/dss/dsshome1/05/di38kid/data/results/runs/{self.name}/log')
        # self.writer = summary.create_file_writer('tensorboard/' + self.name)
        self.loss_type = None

    def setup_optimizer(
            self,
            optimizer: Literal['Adam'] = 'Adam',
            scheduler: Literal['ReduceLROnPlateau', 'CyclicLR'] = 'ReduceLROnPlateau',
            param_groups: List[List[str]] = [],
            initial_lr: Union[float, List] = 0.00001,
            lr_factor: float = 0.1,
            lr_patience: int = 10,
            lr_threshold: float = 0.1,
            threshold_mode: Literal['abs', 'rel'] = 'abs'
    ):
        optimizer_kwargs = []
        scheduler_kwargs = {'base_lr': None, 'max_lr': None}
        param_groups = create_parameter_groups(self.learnable_parameters, num_heliostats=self.heliostat_field.number_of_heliostats)
        self.optimizer = torch.optim.Adam(
            param_groups,
        )
        base_lrs = [group['lr'] for group in param_groups]
        # Define max learning rates (e.g., 10x the base rate)
        max_lrs = [base_lr * 10 for base_lr in base_lrs]

        # Create cyclic learning rate scheduler
        ''' self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=base_lrs,  # List of base LRs for each parameter group
            max_lr=max_lrs,    # List of max LRs for each parameter group
            step_size_up=500,  # Steps in the increasing phase
            mode='triangular2',  # Learning rate policy
            cycle_momentum=False  # Don't cycle momentum when using Adam/AdamW
        ) '''
        
        ''' if len(param_groups) == 0:
            optimizer_kwargs.append({'params': self.learnable_parameters.parameters(), 'lr': initial_lr[0]}) 
        if isinstance(optimizer, str):
            if optimizer == 'Adam':
                if param_groups is None:
                    optimizer_kwargs.append({'params': self.learnable_parameters.parameters(), 'lr': initial_lr[0]})
                    scheduler_kwargs['base_lr'] = initial_lr[0] / 10
                    scheduler_kwargs['max_lr'] = initial_lr[0]
                else:
                    for i, param_group in enumerate(param_groups):
                        params = [self.learnable_parameters[param] for param in param_group]
                        optimizer_kwargs.append({'params': params, 'lr': initial_lr[i]})
                    scheduler_kwargs['base_lr'] = [lr / 10 for lr in initial_lr]
                    scheduler_kwargs['max_lr'] = [lr for lr in initial_lr]
                self.optimizer = torch.optim.Adam(optimizer_kwargs)
            elif optimizer == 'AdamW':
                self.optimiizer = torch.optim.AdamW()
            else:
                raise ValueError(
                    f"Optimizer name not found, change name or include new optimizer."
                    )
        else:
            raise ValueError(f"Optimizer must be given as str.") '''
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
            do_raytracing: bool = True,
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
        pred_reflection_directions = torch.zeros((num_samples, num_heliostats, 4), device=device)
        pred_bitmaps = torch.zeros((num_samples, num_heliostats, 256, 256), device=device)
        diff_bitmaps = torch.zeros_like(pred_bitmaps).to(device=device)
        
        import time
        start = time.time()
        for sample, data in enumerate(field_batch):
            
            incident_ray_directions = data['incident_rays']
            target_area_names = data['receiver_targets']
            target_areas = [self.scenario.get_target_area(name) for name in target_area_names]
            
            self.heliostat_field.align_surfaces_with_motor_positions(
                motor_positions=data['motor_positions'],
                device=device
            )
            sample_orientations = self.heliostat_field.rigid_body_kinematic.orientations
            target_reflection_directions[sample] = (data['flux_centers'] - sample_orientations[:, 0:4, 3])
            # target_reflection_directions[sample] = raytracing_utils.reflect(
            #     incident_ray_directions, data['surface_normals']
            #     )
            # print("Orientations: Surface origins:")
            # print(sample_orientations[:, :, 3])
            
            if do_raytracing:
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
                # print("Orientations: Surface normals")
                # print(sample_orientations[:, 0:4, 2])
                concentrator_normals = concentrator_normals / torch.norm(concentrator_normals, dim=1, keepdim=True)  # normalize
                for n_heliostat in range(num_heliostats):
                    pred_reflection_directions[sample, n_heliostat] = raytracing_utils.reflect(
                        incident_ray_directions[n_heliostat], concentrator_normals[n_heliostat]
                    )
                
        end = time.time()
        # print('forward loop took:', end-start)
        
        if do_raytracing:
            # rescale prediction bitmpas
            target_bitmaps = torch.stack([sample['flux_images'] for sample in field_batch]).to(device)
            pred_bitmaps = self.rescale_flux_bitmaps(pred_bitmaps, target_bitmaps)
            diff_bitmaps = target_bitmaps - pred_bitmaps
            
            contour_bitmaps = contour_difference(predictions=pred_bitmaps, 
                                                targets=target_bitmaps, 
                                                threshold=0.5, 
                                                sharpness=20.0)
        
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
        
        alignment_errors = self.calculate_angles_mrad(
                pred_reflection_directions, 
                target_reflection_directions
            )
        
        # print(f"\talignment errors:")
        # for sample, field_errors in enumerate(alignment_errors.tolist()):
        #     print(f'\t\t[{sample}]: {field_errors}')
        # print()
        
        avg_per_heliostat = alignment_errors.mean(dim=0)
        print(f"\taverage errors: {avg_per_heliostat.tolist()}")
        
        avg_of_field = avg_per_heliostat.mean()
        print(f"\taverage over whole field: {avg_of_field.item()}")

        flat_errors = alignment_errors.flatten()
        median_error = torch.median(flat_errors)
        # print(f"\tmedian over whole field: {median_error.item()}")

        if do_raytracing:
            mae = torch.nn.L1Loss()
            # mean absolute aligment eror [mrad]
            maae = mae(alignment_errors, torch.zeros_like(alignment_errors))
            
            mse = torch.nn.MSELoss()
            # mean squared alignment error [mrad **2]
            msae = mse(alignment_errors, torch.zeros_like(alignment_errors)) 
            print(f"\tMSAE: {msae.item()}")
            
            # Calculate per-heliostat MAE on bitmaps
            # mspe_per_heliostat = torch.mean(torch.square(pred_bitmaps - target_bitmaps), dim=(0, 2, 3))
            # Get overall MAE from per-heliostat values if needed
            # overall_mspe = torch.mean(mspe_per_heliostat)
            
            mape = diff_bitmaps.abs().mean()
            mspe = mse(pred_bitmaps, target_bitmaps)
            # print(f"\tMSPE: {mspe_per_heliostat.tolist()}")
            # print(f"\tField MSPE: {overall_mspe.item()}")
            # print(f"\tField MSPE: {mspe.item()}")
            
            # chamfer_loss = chamfer_distance(flux_bitmap_pred[0], flux_bitmaps[0])
            ssim_fnc = StructuralSimilarityIndexMeasure().to(device)
            ssim = ssim_fnc(pred_bitmaps.reshape(-1, 256, 256).unsqueeze(1), 
                            target_bitmaps.reshape(-1, 256, 256).unsqueeze(1))
            print(f"\tSSIM: {ssim.item()}")
            
            chamfer_distances = chamfer_distance_batch_optimized(pred_bitmaps.reshape(-1, 256, 256), 
                                                                 target_bitmaps.reshape(-1, 256, 256))
            non_zero_mask = chamfer_distances != 0.0
            non_zero_count = non_zero_mask.sum()
            # Calculate mean of non-zero values
            mean_chd = chamfer_distances[non_zero_mask].sum() / non_zero_count
            # mean_chd = chamfer_distances.mean()
            
            # loss_type = '(1 - Structural Similarity Index)'
            # chamfer_distances = chamfer_distance_batch(pred_bitmaps.reshape(-1, 256, 256), 
            #                                            target_bitmaps.reshape(-1, 256, 256))
            # chd = chamfer_distances.mean()
            
            print(f"\tCHD: {mean_chd.item()}")
            
            # Calculate combined loss from alignment MSE and flux MSE.
            alpha = 0.8
            # loss = alpha * msae + (1 - alpha) * mspe
            # TODO: Make alpha learnable https://github.com/Helmholtz-AI-Energy/propulate/blob/main/tutorials/nm_example.py
            loss = alpha * mean_chd + (1-alpha) * maae 
            
            self.writer.add_scalar(f"Loss/MeanCHD/{mode}", mean_chd.item(), epoch)
            self.writer.add_scalar(f"Loss/MAAE/{mode}", maae.item(), epoch)
            self.writer.add_scalar(f"Loss/MSAE/{mode}", msae.item(), epoch)
            self.writer.add_scalar(f"Loss/MeanSSIM/{mode}", ssim.item(), epoch)
            self.writer.add_scalar(f"Loss/MAPE/{mode}", mape.item(), epoch)
            self.writer.add_scalar(f"Loss/MSPE/{mode}", mspe.item(), epoch)
            self.writer.add_scalar(f"Loss/MeanCHD+MAAE/{mode}", loss.item(), epoch)

        else:
            mae = torch.nn.L1Loss()
            mse = torch.nn.MSELoss()
            loss = mse(alignment_errors, torch.zeros_like(alignment_errors))
            loss_type = 'MSE (Alignment Errors)'

        self.writer.add_scalar(f"AlignmentErrors_mrad/{mode}/Average", avg_of_field.item(), epoch)
        self.writer.add_scalar(f"AlignmentErrors_mrad/{mode}/Median", median_error.item(), epoch)
        
        for n_heliostat, (heliostat_id, calib_ids) in enumerate(heliostats_and_calib_ids.items()):
            self.writer.add_scalar(f"AlignmentErrors_mrad/Avg/{mode}/{heliostat_id}",
                                   avg_per_heliostat.tolist()[n_heliostat],
                                   epoch)
            for n_sample, (calib_id, error) in enumerate(zip(calib_ids, alignment_errors[:, n_heliostat].tolist())):
                self.writer.add_scalar(f"AlignmentErrors_mrad/{mode}/{heliostat_id}/{calib_id}", 
                                       error, 
                                       epoch)
                if do_raytracing:
                    self.writer.add_image(f"FluxPredictions/{mode}/{heliostat_id}/{calib_id}", 
                                        pred_bitmaps[n_sample, n_heliostat, :, :].cpu().detach(), 
                                        epoch, 
                                        dataformats='HW')
                    if epoch == 1:
                        self.writer.add_image(f"FluxGroundtruths/{mode}/{heliostat_id}/{calib_id}", 
                                            target_bitmaps[n_sample, n_heliostat, :, :].cpu().detach(), 
                                            epoch, 
                                            dataformats='HW')
                    self.writer.add_image(f"FluxDiffs/{mode}/{heliostat_id}/{calib_id}", 
                                        diff_bitmaps[n_sample, n_heliostat, :, :].cpu().detach(), 
                                        epoch, 
                                        dataformats='HW')
                    self.writer.add_image(f"ContourDiffs/{mode}/{heliostat_id}/{calib_id}", 
                                        contour_bitmaps[n_sample, n_heliostat, :, :].cpu().detach(), 
                                        epoch, 
                                        dataformats='HW')
                    ssim = ssim_fnc(pred_bitmaps[n_sample, n_heliostat, :, :].unsqueeze(0).unsqueeze(0),
                                    target_bitmaps[n_sample, n_heliostat, :, :].unsqueeze(0).unsqueeze(0))
                    self.writer.add_scalar(f"SSIM/{mode}/{heliostat_id}/{calib_id}", 
                                        ssim.item(), 
                                        epoch)
                    mspe = mse(pred_bitmaps[n_sample, n_heliostat, :, :].unsqueeze(0).unsqueeze(0),
                            target_bitmaps[n_sample, n_heliostat, :, :].unsqueeze(0).unsqueeze(0))
                    self.writer.add_scalar(f"MSPE/{mode}/{heliostat_id}/{calib_id}", 
                                        mspe.mean().item(), 
                                        epoch)
                    chd = chamfer_distance_batch(pred_bitmaps[n_sample, n_heliostat, :, :].unsqueeze(0),
                                                target_bitmaps[n_sample, n_heliostat, :, :].unsqueeze(0))
                    self.writer.add_scalar(f"CHD/{mode}/{heliostat_id}/{calib_id}", 
                                        chd.mean().item(), 
                                        epoch)
                    
        return alignment_errors, loss

    @staticmethod
    def calculate_angles_mrad(
            v1: torch.Tensor,
            v2: torch.Tensor,
            epsilon: float = 1e-8
            ) -> torch.Tensor:
        # in case v1 or v2 have shape (4,), bring to (1, 4)
        assert v1.shape == v2.shape, 'Given Tensors must have identical shapes.'
        while v1.dim() < 3:  
            v1 = v1.unsqueeze(0)  # bring to [B, H, 4] 
            v2 = v2.unsqueeze(0) 

        m1 = torch.norm(v1, dim=-1)
        m2 = torch.norm(v2, dim=-1)
        dot_products = torch.sum(v1 * v2, dim=-1)
        cos_angles = dot_products / (m1 * m2 + epsilon)
        angles_rad = torch.acos(
            torch.clamp(cos_angles, min= -1.0 + 1e-7, max= 1.0 - 1e-7)
        )
        return angles_rad * 1000

    def calibrate(
            self,
            split_type: str,
            split_sizes: Tuple[int, int],
            num_epochs: int = 30,
            log_steps: int = 10,
            tolerance: float = 0.0,
            seed: int = random_seed,
            exclude_params: List[str] = (),
            use_raytracing: bool = True,
            run_config: dict = None,
            device: Union[torch.device, str] = "cuda"
    ):

        split_df = self.calibration_data_loader.splits[split_type][split_sizes]
        
        for param in exclude_params:
            try:
                self.learnable_parameters[param].requires_grad = False
            except KeyError:
                raise ValueError(f"Parameter {param} was not found in the learnable parameters.")

        log.info(f"Starting Calibration with motor positions...")
        epoch = 1
        avg_valid_error = torch.inf
        while epoch <= num_epochs and avg_valid_error > tolerance:
            self.train()
            self.optimizer.zero_grad()

            train_df = split_df[split_df['Split'] == 'train']
            train_dict = {
                heliostat_id: train_df.loc[train_df['HeliostatId'] == heliostat_id].index.tolist()
                for heliostat_id in split_df.loc[split_df['Split'] == 'train', 'HeliostatId'].unique()
            }
            
            train_alignment_errors, train_loss = self.forward(
                mode='Train', epoch=epoch,
                heliostats_and_calib_ids=train_dict,
                do_raytracing=use_raytracing,
                do_plotting=False,
                device=device
            )
            train_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.learnable_parameters.parameters(), max_norm=0.5)            
            
            check_for_nan_grad(self.learnable_parameters)

            self.optimizer.step()

            self.eval()
            with torch.no_grad():
                val_df = split_df[split_df['Split'] == 'validation']
                val_dict = {
                    heliostat_id: val_df.loc[val_df['HeliostatId'] == heliostat_id].index.tolist()
                    for heliostat_id in split_df.loc[split_df['Split'] == 'validation', 'HeliostatId'].unique()
                }
            
                
                valid_alignment_errors, valid_loss = self.forward(
                    mode='Validation', epoch=epoch,
                    heliostats_and_calib_ids=val_dict,
                    do_raytracing=use_raytracing,
                    do_plotting=False,
                    device=device
                )
            self.scheduler.step(valid_loss)

            avg_train_error = train_alignment_errors.mean()
            avg_valid_error = valid_alignment_errors.mean()
            
            self._log_parameters(epoch=epoch)
            
            # self.writer.add_scalar("Loss/Train", train_loss.cpu().detach().numpy(), epoch)
            # self.writer.add_scalar("Loss/Valid", valid_loss.cpu().detach().numpy(), epoch)
            # self.writer.add_scalar("AvgError/Train", avg_train_error.item(), epoch)
            # self.writer.add_scalar("AvgError/Valid", avg_valid_error.item(), epoch)
            # self.writer.add_scalar("MedError/Train", avg_train_error.flatten().median().item(), epoch)
            # self.writer.add_scalar("MedError/Valid", avg_valid_error.flatten().median().item(), epoch)
            

            # for calib_id, error in zip(train_ids, train_alignment_errors):
            #     self.writer.add_scalar(f"Errors/Train/{calib_id}", error.cpu().detach().numpy(), epoch)

            # for calib_id, error in zip(valid_ids, valid_alignment_errors):
            #     self.writer.add_scalar(f"Errors/Valid/{calib_id}", error.cpu().detach().numpy(), epoch)
                      
            # TODO: Log Raytracer Flux Bitmaps to Tensorboard.
            # summary.image("FluxBitmaps/id", flux_bitmap, epoch)

            avg_train_error = train_alignment_errors.mean()
            avg_valid_error = valid_alignment_errors.mean()

            if epoch % log_steps == 0:
                log.info(f"Epoch: {epoch} of {num_epochs} max, "
                         f"Train / Valid Loss: {train_loss.item():.4f} / {valid_loss.item():.4f}, "
                         f"Avg Train / Valid Error (mrad): {avg_train_error.item():.2f} / {avg_valid_error.item():.2f}, "
                         # f"LR: {self.scheduler.get_last_lr()}"
                         )

            epoch += 1
        
        test_df = split_df[split_df['Split'] == 'test']
        test_dict = {
            heliostat_id: test_df.loc[test_df['HeliostatId'] == heliostat_id].index.tolist()
            for heliostat_id in split_df.loc[split_df['Split'] == 'test', 'HeliostatId'].unique()
        }
        self.evaluate_model_fit(test_dict, num_epochs, use_raytracing=use_raytracing, device=device)

    def evaluate_model_fit(
            self,
            evaluate_dict: dict[str, list[int]],
            epoch: int,
            use_raytracing: bool = True,
            do_plotting : bool = False,
            device: Union[torch.device, str] = "cuda"
    ):
        self.eval()
        with torch.no_grad():
            test_alignment_errors, test_loss = self.forward(
                    mode='Test', epoch=epoch,
                    heliostats_and_calib_ids=evaluate_dict,
                    do_raytracing=use_raytracing,
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
            obj = self.learnable_parameters
            
        if isinstance(obj, torch.nn.Parameter):
            self.writer.add_histogram(f"Parameters/{name}/{self.heliostat_field.all_heliostat_names[heliostat_idx]}",
                                        obj.cpu().detach().numpy(),
                                        epoch)
            
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


