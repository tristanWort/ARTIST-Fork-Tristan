import torch
import os
import sys
import time
import logging
import json
import h5py
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Optional, Union
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 
from artist.util import config_dictionary, paint_loader
from artist.util.configuration_classes import (LightSourceConfig, LightSourceListConfig,)
from artist.util.scenario_generator import ScenarioGenerator
from artist.util.scenario import Scenario
from artist.scene.light_source_array import LightSourceArray
from artist.scene.sun import Sun
from artist.field.tower_target_area_array import TargetAreaArray
from artist.field.heliostat_field import HeliostatField
from artist.raytracing.heliostat_tracing import HeliostatRayTracer

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, repo_path)
from HeliOptiCal.data_generation.generate_scenario import build_heliostat_file_list
from HeliOptiCal.utils import my_config_dict, util_dataset
from HeliOptiCal.utils.util_simulate import align_and_raytrace
from HeliOptiCal.utils.util import normalize_images
from HeliOptiCal.image_losses.image_loss import sdf_loss, dice_loss


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
# A logger for the FixRaytracer class.

"""
A class which emulates the ARTIST environment. 
Attempts to optimize the hyperparameters used for Raytracing (such as sun distribution parameters)
for finding parameters that give flux predictions which are closer to the measured PAINT data.
"""

save_bitmaps_dir ="/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/experiments/bitmaps"


def save_bitmap(bitmap, save_dir, save_as, width_in_pixels=512, height_in_pixels=512):
    dpi = 100  # Can be any value
    width_in_inches = width_in_pixels / dpi
    height_in_inches = height_in_pixels / dpi
    plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
    plt.imshow(bitmap.cpu().detach(), cmap="gray")
    plt.axis("off")  # Hides both x and y axes
    plt.savefig(Path(save_dir)/save_as, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()

class FixRaytracer(torch.nn.Module):
    
    def __init__(self, helio_measurements_dir: Union[str, Path], device: Union[str, torch.device]="cuda:0"):
        
        super().__init__()
        log.info("Iniatiating FixRaytracer instance!")
        self.helio_measurements_dir = helio_measurements_dir
        self.device = torch.device(device)
        self.scenario_path = Path("/dss/dsshome1/05/di38kid/data/scenarios/20250525_scenario_20_heliostats")
        self.optimize_parameters = self.initiate_parameters()
        
        # Prepare and generate a scenario
        self.power_plant_config, self.target_area_list_config = self.load_power_plant_n_target_config()
        self.receiver_area = self.get_receiver() 
        self.heliostat_list_config, self.prototype_config = None, None
        self.light_source_list_config = None
        self.number_of_rays = 10
        
        self.scenario = None    
        self.raytracer = None
        
        self.data_batch = []
    
    def initiate_parameters(self, mean=0.0, cov=2 * 4.3681e-6):
        device = self.device
        optimize_parameters = torch.nn.ParameterDict()
        optimize_parameters['light_source_mean'] = torch.nn.Parameter(
            torch.tensor([mean, mean], dtype=torch.float64, device=self.device),
                requires_grad=True
            )
        optimize_parameters['light_source_covariance'] = torch.nn.Parameter(
            torch.tensor([[cov, 0], [0, cov]], dtype=torch.float64, device=self.device, 
                         requires_grad=True)
            )
        
        self.optimize_parameters = optimize_parameters
        return optimize_parameters
    
    def load_power_plant_n_target_config(self, tower_file_path: Union[str, Path]=None):
        
        if tower_file_path is None:
            tower_file_path = Path("/dss/dsshome1/05/di38kid/data/paint/WRI1030197-tower-measurements.json")
        log.info("Loading configurations for power plant and target areas...")
        power_plant_config, target_area_list_config = (
            paint_loader.extract_paint_tower_measurements(tower_file_path, device=self.device)
            )
        self.power_plant_config = power_plant_config
        self.target_area_list_config = target_area_list_config
        return power_plant_config, target_area_list_config
        
    def load_heliostat_list_config(self, heliostat_config_files: List[Tuple]):
        
        log.info(f"Loading configurations for {len(heliostat_config_files)} Heliostats...")
        heliostat_list_config, prototype_config = paint_loader.extract_paint_heliostats(
            heliostat_and_deflectometry_paths=heliostat_config_files,
            power_plant_position=self.power_plant_config.power_plant_position,
            aim_point=self.receiver_area.center,
            max_epochs_for_surface_training=10000,
            device=self.device)
        self.heliostat_list_config = heliostat_list_config
        self.prototype_config = prototype_config
        return heliostat_list_config, prototype_config
    
    def load_light_source_config(self, mean=0.0, covariance=2 * 4.3681e-6, 
                                 number_of_rays=10, 
                                 distribution_type="normal"):

        if self.optimize_parameters is None:
            self.optimize_parameters = self.initiate_parameters()
            
        log.info(f"Loading configurations of one light source sun...")
        light_source1_config = LightSourceConfig(light_source_key="sun_1",
                                                 light_source_type=config_dictionary.sun_key,
                                                 number_of_rays=number_of_rays,
                                                 distribution_type=distribution_type,
                                                 mean=mean,
                                                 covariance=covariance)
        light_source_list_config = LightSourceListConfig(light_source_list=[light_source1_config])
        self.light_source_list_config = light_source_list_config
        return light_source_list_config
    
    def get_receiver(self, target_area_list_config=None):
        
        if target_area_list_config is None:
            target_area_list_config = self.target_area_list_config
            
        target_areas = [target_area for target_area in target_area_list_config.target_area_list
                         if target_area.target_area_key == config_dictionary.target_area_reveicer]
        self.receiver_area = target_areas[0]
        return target_areas[0]
    
    def generate_h5_scenario_file(self, scenario_path: Optional[Path]=None):
        
        if scenario_path is None:
            scenario_path = self.scenario_path
            
        if self.power_plant_config is None:
            self.self.load_power_plant_n_target_config()
            
        if self.heliostat_list_config is None:
            heliostat_field_list = build_heliostat_file_list(self.helio_measurements_dir)
            self.load_heliostat_list_config(heliostat_field_list) 
            
        if self.light_source_list_config is None:
            self.load_light_source_config()
        
        log.info(f"Generating one scenario at {scenario_path}...")
        scenario_generator = ScenarioGenerator(
            file_path=scenario_path,
            power_plant_config=self.power_plant_config,
            target_area_list_config=self.target_area_list_config,
            light_source_list_config=self.light_source_list_config,
            prototype_config=self.prototype_config,
            heliostat_list_config=self.heliostat_list_config)
        scenario_generator.generate_scenario()
        
    def load_scenario(self, scenario_h5_path: Optional[Path]=None):
        
        if scenario_h5_path is None:
            scenario_h5_path = self.scenario_path
        
        if self.light_source_list_config is None:
            self.load_light_source_config()
        
        log.info(f"Loading one scenario from {scenario_h5_path}...")
        with h5py.File("/dss/dsshome1/05/di38kid/data/scenarios/20250525_scenario_20_heliostats.h5", "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(scenario_file=scenario_file, device=self.device)   
        return scenario

    def infuse_parameters(self, scenario=None, distribution_parameters: Dict=None):

        if scenario is None:
            if self.scenario is None:
                self.scenario = self.load_scenario()  
            scenario = self.scenario
        log.info("Infusing light source distribution parameters...")
        
        distribution = self.build_distribution()
        sun = scenario.light_sources.light_source_list[0]
        
        sun.distribution = distribution
        return scenario
    
    def build_distribution(self):
        
        # optimize_parameters = torch.nn.ParameterDict()
        device = self.device
        
        distribution = torch.distributions.MultivariateNormal(self.optimize_parameters['light_source_mean'], 
                                                              self.optimize_parameters['light_source_covariance'])
        return distribution
        
    def init_raytracer(self, scenario=None, bitmap_resolution=(256, 256)):
        
        if scenario is None:
            scenario = self.scenario
            
        # Before raytracer setup, Heliostat field needs to be aligned
        heliostat_field = scenario.heliostat_field
        heliostat_field.align_surfaces_with_incident_ray_direction(
            torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device),
            device=self.device)
        
        raytracer = HeliostatRayTracer(scenario=scenario,
                                       world_size=1,
                                       rank=0,
                                       batch_size=1,
                                       random_seed=42,
                                       bitmap_resolution_e=bitmap_resolution[0],
                                       bitmap_resolution_u=bitmap_resolution[1])
        return raytracer
        
    def setup(self, scenario_h5_path=None):
        
        if scenario_h5_path is None:
            scenario_h5_path = self.scenario_path
        
        log.info("Begin setup...")
        scenario = self.load_scenario(scenario_h5_path)
        # self.optimize_parameters = self.initiate_parameters()
        self.scenario = self.infuse_parameters(scenario)
        self.raytracer = self.init_raytracer(self.scenario)
        
    def load_calibration_data(self, helio_to_calib_map: Dict[str, List]):
        
        heliostat_names = self.scenario.heliostat_field.all_heliostat_names
        assert heliostat_names == list(helio_to_calib_map.keys()), "unequal heliostat names!"
        
        batch_size = len(next(iter(helio_to_calib_map.values())))
        data_batch = [
            {
                my_config_dict.field_sample_calibration_ids: [],
                my_config_dict.field_sample_sun_elevations: [],
                my_config_dict.field_sample_sun_azimuths: [],
                my_config_dict.field_sample_incident_rays: [],
                my_config_dict.field_sample_target_names: [],
                my_config_dict.field_sample_flux_centers: [],
                my_config_dict.field_sample_flux_images: []
            } for _ in range(batch_size)
            ]
        
        json_suffix = "-calibration-properties.json"
        img_suffix = "-flux.png"
        for h, name in enumerate(heliostat_names):
            log.info(f"Loading calibration data for {name}")
            calib_ids = helio_to_calib_map[name]
            assert len(calib_ids) == batch_size
            
            calibration_directory = Path(self.helio_measurements_dir) / f"{name}/Calibration"
            if not os.path.exists(calibration_directory):
                raise FileNotFoundError(f"Calibration folder not found at path: "
                                        f"{calibration_directory}")
            for b, calib_id in enumerate(calib_ids):
                
                properties_file = [os.path.join(calibration_directory, f)
                                   for f in os.listdir(calibration_directory) 
                                   if f.endswith(f"{calib_id}{json_suffix}")][0]

                # Extract data from JSON properties file       
                (
                    target_name, flux_center, azimuth, elevation, sun_position, motor_positions
                 ) = util_dataset.extract_paint_calibration_data(Path(properties_file),
                                                                  self.scenario.power_plant_position,
                                                                  device=self.device)
                incident_ray = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(self.device) - sun_position
                
                flux_path = [os.path.join(calibration_directory, f)
                            for f in os.listdir(calibration_directory) 
                            if f.endswith(f"{calib_id}{img_suffix}")][0]
                
                # Load flux image
                flux_image = Image.open(flux_path)
                transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((256, 256)),
                    transforms.ToTensor() 
                    ])
                flux_image = transform(flux_image).squeeze(0)
                
                data_batch[b][my_config_dict.field_sample_calibration_ids].append(calib_id)
                data_batch[b][my_config_dict.field_sample_sun_elevations].append(elevation)
                data_batch[b][my_config_dict.field_sample_sun_azimuths].append(azimuth)
                data_batch[b][my_config_dict.field_sample_incident_rays].append(incident_ray)
                data_batch[b][my_config_dict.field_sample_target_names].append(target_name)
                data_batch[b][my_config_dict.field_sample_flux_centers].append(flux_center)
                data_batch[b][my_config_dict.field_sample_flux_images].append(flux_image)
        
        self.data_batch = data_batch
        log.info("Loading of calibration data complete!")
    
    def forward(self, data_batch: List[Dict]=None):   

        if data_batch is None:
            data_batch = self.data_batch
        
        self.setup()
        raytracer = self.raytracer
        
        heliostat_field = self.scenario.heliostat_field 
        kinematic = heliostat_field.rigid_body_kinematic
        
        B = len(data_batch)
        H = len(data_batch[0][my_config_dict.field_sample_calibration_ids])
        
        # all_bitmaps = torch.zeros([B, H, 256, 256], dtype=torch.float32, device=self.device)
        all_bitmaps = []
        
        log.info("Raytracing...")
        for b, data in enumerate(data_batch):
            
            # Get the required data
            calibration_ids = data[my_config_dict.field_sample_calibration_ids]
            sun_elevations = data[my_config_dict.field_sample_sun_elevations]
            sun_azimuths = data[my_config_dict.field_sample_sun_azimuths]
            incident_ray_directions = data[my_config_dict.field_sample_incident_rays]
            target_area_names = data[my_config_dict.field_sample_target_names]
            target_areas = [self.scenario.get_target_area(name) for name in target_area_names]
            
            # Use alignment based on incident rays, to get new motor positions    
            aim_points = torch.stack(data[my_config_dict.field_sample_flux_centers]) 
            kinematic.aim_points = aim_points
            incident_ray_directions = torch.stack(incident_ray_directions) 
              
            heliostat_field.align_surfaces_with_incident_ray_direction(
                incident_ray_direction=incident_ray_directions,
                round_motor_pos=False,
                device=self.device)
            motor_positions = kinematic.motor_positions
            
            bitmaps = align_and_raytrace(scenario=self.scenario,
                                         incident_ray_directions=incident_ray_directions,
                                         target_areas=target_areas,
                                         align_with_motor_positions=True,
                                         motor_positions=motor_positions,
                                         raytracer=raytracer,
                                         device=self.device)
            
            # bitmaps = align_and_raytrace(scenario=self.scenario,
            #                              incident_ray_directions=incident_ray_directions,
            #                              target_areas=target_areas,
            #                              aim_points=aim_points,
            #                              align_with_motor_positions=False,
            #                              raytracer=raytracer,
            #                              device=self.device)
            all_bitmaps.append(bitmaps)
        return torch.stack(all_bitmaps)
    
    def optimize(self, n_epochs: int=1000, learning_rate: float=1e-5):
        
        self.setup()
        optimizer = torch.optim.Adam(self.optimize_parameters.parameters(), lr=learning_rate)
        helio_to_calib_map = self.get_map(5)
        data_batch = self.load_calibration_data(helio_to_calib_map)
        
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(n_epochs):
                
                pred_bitmaps = self.forward(data_batch)
                if epoch % 10 == 0:
                    save_bitmap(pred_bitmaps[0, 0], save_bitmaps_dir, f"pred_0_8_{epoch}.png")
                    
                loss = self.evaluate(pred_bitmaps)
                optimizer.zero_grad()
                # torch.autograd.set_detect_anomaly(True)

                self.optimize_parameters['light_source_mean'].retain_grad()
                self.optimize_parameters['light_source_covariance'].retain_grad()
                loss.backward()
                print(self.optimize_parameters['light_source_mean'].grad)
                print(self.optimize_parameters['light_source_covariance'].grad)
                
                optimizer.step()
                mean = self.optimize_parameters['light_source_mean'].detach().cpu().tolist()
                cov = self.optimize_parameters['light_source_covariance'].detach().cpu().tolist()
                log.info(f"[Epoch {epoch} / {n_epochs}] [Loss: {loss.item():.8f}] "
                        f"[Sun mean {mean} and sun cov {cov}]")
    
    def evaluate(self, pred_bitmaps):        
        
        B, H = pred_bitmaps.shape[0], pred_bitmaps.shape[1]
        device = pred_bitmaps.device
        
        true_bitmaps = torch.stack(
            [torch.stack(data_batch[my_config_dict.field_sample_flux_images])
             for data_batch in self.data_batch]
            )
        assert pred_bitmaps.shape == true_bitmaps.shape, f"unequal shapes: {pred}"
        
        mse = torch.nn.MSELoss()
        
        losses = []
        norm_pred = []
        norm_true = []
        
        for h in range(pred_bitmaps.shape[1]):
            
            norm_pred = normalize_images(pred_bitmaps[:, h])
            with torch.no_grad():
                norm_true = normalize_images(true_bitmaps[:, h])
            
            losses.append(mse(norm_pred, norm_true))

        save_bitmap(norm_true[0], save_bitmaps_dir, f"norm_true_0_0.png")
        
        loss = torch.stack(losses).mean()
        return loss
            
    @staticmethod
    def get_map(size: int=5):
        
        # all_helio_to_calib_map = {
        #     "AA39": [163373],
        #     "AM35": [68534]
        #     }
        
        all_helio_to_calib_map = {
            "AA26": [61918, 63627, 133730, 152147, 190045],
            "AA39": [163373, 169244, 192811, 199277, 219162],
            "AE32": [61656, 67960, 71199, 72531, 183551],
            "AF46": [66322, 71030, 78960, 95821, 139152],
            "AH30": [66154, 66866, 74369, 87842, 115133],
            "AM35": [68534, 73675, 87786, 111295, 155687],
            "AO51": [59000, 66339, 71750, 86085, 106174],
            "AP30": [64910, 67423, 138731, 145335, 193278],
            "AP41": [110492, 121271, 137304, 147674, 192662],
            "AX39": [74393, 91205, 113541, 178239, 215399],
            "AZ52": [87621, 124874, 145189, 168343, 230095],
            "BA41": [72450, 95184, 120445, 137846, 181205],
            "BA73": [65505, 73159, 88921, 107046, 180910],
            "BB37": [62614, 68794, 101226, 149335, 214070],
            "BC28": [82081, 86516, 186458, 226033, 235651],
            "BC34": [82418, 86487, 98096, 126526, 171320],
            "BC62": [82546, 99383, 104677, 106569, 137171],
            "BD40": [82056, 96297, 110845, 140361, 225663],
            "BF27": [119068, 120184, 142855, 188089, 247031],
            "BG50": [190146, 199077, 200582, 268478, 274322],
        }
        
        helio_to_calib_map = {h_name: [] for h_name in all_helio_to_calib_map}
        for h_name, all_calib_ids in all_helio_to_calib_map.items():
            helio_to_calib_map[h_name] = all_calib_ids[:size]
        return helio_to_calib_map

    

if __name__ == '__main__':
    
    helio_dir = "//dss/dsshome1/05/di38kid/data/paint/selected_20"
    heliostat_field_list = build_heliostat_file_list(helio_dir)
    fixer = FixRaytracer(helio_dir, device="cpu")
    
    # fixer.load_heliostat_list_config(heliostat_field_list)
    # fixer.load_light_source_config()
    # fixer.generate_h5_scenario_file()
    
    # scenario = fixer.setup()
    
    # fixer.load_calibration_data(helio_to_calib_map)
    output = fixer.optimize(100, learning_rate=1e-6)
    # print()

    