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
repo_path = os.path.abspath(os.path.dirname('/jump/tw/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 
from artist.scenario import Scenario
from artist.raytracing import raytracing_utils
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util import utils

from calibration_dataset import CalibrationDataLoader
from util import get_rigid_body_kinematic_parameters_from_scenario
from image_loss import chamfer_distance_batch


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
                    load_flux_images=True,
                    device=device
                    )
        self.calibration_data_loader.sun_positions_splits(config=config['splits'])

        self.learnable_parameters = (
            get_rigid_body_kinematic_parameters_from_scenario(
                kinematic=self.heliostat_field.rigid_body_kinematic
            )
        )
        total_parameters = sum(param.numel() for param in self.learnable_parameters.values())
        log.info(f"Total number of parameters: {total_parameters}")

        self.optimizer, self.scheduler = self.setup_optimizer(optimizer=model_config['optimizer_type'],
                                                              scheduler=model_config['scheduler_type'],
                                                              # param_groups=model_config['param_groups'],  #TODO: complete model config
                                                              initial_lr=model_config['init_learning_rate'],
                                                              )

        # self.writer = SummaryWriter(log_dir=f'/jump/tw/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/tensorboard/{self.name}')
        # self.writer = summary.create_file_writer('tensorboard/' + self.name)
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
        if isinstance(optimizer, str):
            if optimizer == 'Adam':
                if param_groups is None:
                    optimizer_kwargs.append({'params': self.learnable_parameters.values(), 'lr': initial_lr})
                    scheduler_kwargs['base_lr'] = initial_lr[0] / 10
                    scheduler_kwargs['max_lr'] = initial_lr[0]
                else:
                    for i, param_group in enumerate(param_groups):
                        params = [self.learnable_parameters[param] for param in param_group]
                        optimizer_kwargs.append({'params': params, 'lr': initial_lr[i]})
                    scheduler_kwargs['base_lr'] = [lr / 10 for lr in initial_lr]
                    scheduler_kwargs['max_lr'] = [lr for lr in initial_lr]
                self.optimizer = torch.optim.Adam(optimizer_kwargs)
            else:
                raise ValueError(f"Optimizer {optimizer} not included.")
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
            else:
                raise ValueError(f"Scheduler must be of provided as str.")
            log.info(f"Optimizer and Scheduler setup complete.")
            return self.optimizer, self.scheduler

    def forward(
            self,
            heliostats_and_calib_ids: dict[str, List[int]],
            do_raytracing: bool = True,
            do_plotting: bool = False,
            device: Union[torch.device, str] = "cuda"
            ):
        """Generate model output for the next step"""
        device = torch.device(device)
        # self.heliostat_field.is_aligned = True
        
        for heliostat_id, calibration_ids in heliostats_and_calib_ids.items():
            batch = self.calibration_data_loader.get_batch(heliostat_id=heliostat_id, 
                                                           cal_ids=calibration_ids)

            num_samples = len(calibration_ids)
            reflection_directions_calibration = torch.zeros((num_samples, 4), device=device)
            reflection_directions = torch.zeros((num_samples, 4), device=device)
            flux_pred_bitmaps = torch.zeros((num_samples, 256, 256), device=device)
            # diff_flux_bitmaps = torch.zeros((num_samples, 256, 256), device=device)

            chds = torch.empty(num_samples, device=device)
            for sample, calib_id in enumerate(calibration_ids):
                # print(f"\tRaytrace Sample {sample} of {len(calibration_ids)}")
                # Use motor positions to get the concentrator orientation.
                # single_batch = self.calibration_data_loader.get_single_item(calib_id)
                orientation = self.heliostat_field.rigid_body_kinematic.motor_positions_to_orientation(
                    motor_positions=batch['motor_positions'][sample],
                    device=device
                )

                reflection_directions_calibration[sample] = (batch['flux_centers'][sample] - orientation[:, 3])

                # Translate and rotate surface points and surface normals.
                aligned_surface_points = self.heliostat.surface_points @ orientation.T
                aligned_surface_normals = self.heliostat.surface_normals @ orientation.T
                self.heliostat.current_aligned_surface_points = aligned_surface_points
                self.heliostat.current_aligned_surface_normals = aligned_surface_normals


                # Set up profiler
                with torch.profiler.profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True
                )as prof:
                
                    if do_raytracing:
                        # print_gpu_memory_usage(self.device, f"{__name__}: line 267")
                        target_area = self.target_areas[batch['receiver_targets'][sample]]
                        self.raytracer.target_area = target_area
                        # with torch.no_grad():
                        flux_pred_bitmap = self.raytracer.trace_rays(
                            incident_ray_direction=batch['incident_rays'][sample],
                            device=device
                        )
                        # print_gpu_memory_usage(self.device, f"{__name__}: line 275")
                        flux_pred_bitmap = self.raytracer.normalize_bitmap(flux_pred_bitmap)
                        # print_gpu_memory_usage(self.device, f"{__name__}: line 277")
                        # Prediction should match energy content from recorded flux bitmap.
                        scaling_factor = batch['flux_images'][sample].sum() / flux_pred_bitmap.sum()
                        flux_pred_bitmap = scaling_factor * flux_pred_bitmap

                        center_of_mass = utils.get_center_of_mass(
                            bitmap=flux_pred_bitmap,
                            target_center=target_area.center,
                            plane_e=target_area.plane_e,
                            plane_u=target_area.plane_u,
                            device=device
                        )

                        reflection_directions[sample] = (center_of_mass - orientation[:, 3])
                        # Store pixel difference in flux_bitmaps.
                        flux_pred_bitmaps[sample] = flux_pred_bitmap
                        # diff_flux_bitmaps[sample] = (flux_pred_bitmap - batch['flux_image'])

                        # print_gpu_memory_usage(self.device, f"{__name__}: line 295")
                        # chds[sample] = chamfer_distance_batch(flux_pred_bitmap.unsqueeze(0), single_batch['flux_image'].unsqueeze(0)).mean()

                        #TODO: Function does not map the centers properly yet.
                        # if do_plotting:
                        #     center1 = enu_point_to_target_plane(
                        #         enu_point=batch['flux_centers'][sample],
                        #         target_center = target_area.center,
                        #         plane_e = target_area.plane_e,
                        #         plane_u = target_area.plane_u,
                        #     )
                        #     center2 = enu_point_to_target_plane(
                        #         enu_point=center_of_mass,
                        #         target_center=target_area.center,
                        #         plane_e=target_area.plane_e,
                        #         plane_u=target_area.plane_u,
                        #     )
                        #     plot_calibration_result(
                        #         bitmap1=batch['flux_images'][sample],
                        #         bitmap2=flux_pred_bitmap,
                        #         center1=center1,
                        #         center2=center2,
                        #         show_diff=True,
                        #         device=device
                        #     )


                    else:  # No raytracing, calculate error from heliostat orientation.
                        reflection_directions[sample] = raytracing_utils.reflect(
                            -batch['incident_rays'][sample], orientation[:,2]
                        )
                    
                # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=40)) 

            # Calculate the error between both reflection directions.
            alignment_errors = self.calculate_angles_mrad(
                reflection_directions_calibration, reflection_directions
            )

            if do_raytracing:
                mse = torch.nn.MSELoss()
                msae = mse(alignment_errors, torch.zeros_like(alignment_errors))
                
                diff_flux_bitmaps = flux_pred_bitmaps - batch['flux_images']
                mape = diff_flux_bitmaps.abs().mean()
                
                # chamfer_loss = chamfer_distance(flux_bitmap_pred[0], flux_bitmaps[0])
                ssim_fnc = StructuralSimilarityIndexMeasure().to(device)
                ssim = ssim_fnc(flux_pred_bitmaps.unsqueeze(1).to(device), batch['flux_images'].unsqueeze(1).to(device))
                
                # loss_type = '(1 - Structural Similarity Index)'
                chamfer_distances = chamfer_distance_batch(flux_pred_bitmaps, batch['flux_images'])
                chd = chamfer_distances.mean()
                
                print(f"\tMSAE: {msae.item()}")
                print(f"\tMAPE: {mape.item()}")
                print(f"\tSSIM: {ssim.item()}")
                print(f"\tCHD: {chd.item()}")
                loss = msae

            else:
                mse = torch.nn.MSELoss()
                loss = mse(alignment_errors, torch.zeros_like(alignment_errors))
                loss_type = 'MSE (Alignment Errors)'

        return alignment_errors, loss

    @staticmethod
    def calculate_angles_mrad(
            v1: torch.Tensor,
            v2: torch.Tensor,
            epsilon: float = 1e-8
    ) -> torch.Tensor:
        # in case v1 or v2 have shape (4,), bring to (1, 4)
        v1 = v1.unsqueeze(0) if v1.dim() == 1 else v1
        v2 = v2.unsqueeze(0) if v2.dim() == 1 else v2

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
            with_raytracing: bool = True,
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
                heliostats_and_calib_ids=train_dict,
                do_raytracing=with_raytracing,
                do_plotting=False,
                device=device
            )
            train_loss.backward()
            for name, param in self.learnable_parameters.items():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"Parameter {name} has NaN gradient")
            self.optimizer.step()

            self.eval()
            with torch.no_grad():
                valid_alignment_errors, valid_loss = self.forward(
                    calibration_ids=split_df[split_df['Split'] == 'train'].index.tolist(),
                    do_raytracing=with_raytracing,
                    do_plotting=False,
                    device=device
                )
            self.scheduler.step(valid_loss)

            avg_train_error = train_alignment_errors.mean()
            avg_valid_error = valid_alignment_errors.mean()
            
            self.writer.add_scalar("Loss/Train", train_loss.cpu().detach().numpy(), epoch)
            self.writer.add_scalar("Loss/Valid", valid_loss.cpu().detach().numpy(), epoch)
            self.writer.add_scalar("AvgError/Train", avg_train_error.cpu().detach().numpy(), epoch)
            self.writer.add_scalar("AvgError/Valid", avg_valid_error.cpu().detach().numpy(), epoch)

            for calib_id, error in zip(train_ids, train_alignment_errors):
                self.writer.add_scalar(f"Errors/Train/{calib_id}", error.cpu().detach().numpy(), epoch)

            for calib_id, error in zip(valid_ids, valid_alignment_errors):
                self.writer.add_scalar(f"Errors/Valid/{calib_id}", error.cpu().detach().numpy(), epoch)

            for name, param in self.learnable_parameters.items():
                if torch.isnan(param).any():
                    print(f"Parameter {name} is NaN.")
                self.writer.add_histogram(name, param.cpu().detach().numpy(), epoch)

            # TODO: Log Raytracer Flux Bitmaps to Tensorboard.
            # summary.image("FluxBitmaps/id", flux_bitmap, epoch)

            avg_train_error = train_alignment_errors.mean()
            avg_valid_error = valid_alignment_errors.mean()

            if epoch % log_steps == 0:
                log.info(f"Epoch: {epoch} of {num_epochs} max, "
                         f"Train / Valid Loss: {train_loss.item():.4f} / {valid_loss.item():.4f}, "
                         f"Avg Train / Valid Error (mrad): {avg_train_error.item():.2f} / {avg_valid_error.item():.2f}, "
                         f"LR: {self.scheduler.get_last_lr()}")

            epoch += 1

        self.evaluate_model_fit(test_ids, num_epochs, device=device)

    def evaluate_model_fit(
            self,
            calibration_ids: List[str],
            epoch: int,
            device: Union[torch.device, str] = "cuda"
    ):
        self.eval()
        with torch.no_grad():
            test_alignment_errors, test_loss = self.forward(
                calibration_ids=calibration_ids,
                do_raytracing=True,
                do_plotting=False,
                device=device
            )

            avg_test_error = test_alignment_errors.mean()

            # summary.scalar("Loss/Test", test_loss.detach().numpy(), epoch)
            # summary.scalar("AvgError/Test", avg_test_error.detach().numpy(), epoch)

            # for calib_id, error in zip(calibration_ids, test_alignment_errors):
            #     summary.scalar(f"Errors/Valid/{calib_id}", error.detach().numpy(), epoch)

            log.info(f"Test Loss: {test_loss.item():.4f}, "
                     f"Avg Test Error (mrad): {avg_test_error.item():.2f}")

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


