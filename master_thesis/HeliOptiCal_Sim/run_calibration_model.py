import torch
import os
import sys
import h5py
import json

from datetime import datetime
from pathlib import Path

# Add local artist path for raytracing with multiple parallel heliostats.
artist_repo = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, artist_repo)  
from artist.util.scenario import Scenario
from calibration_model import CalibrationModel

# Add local path to modify scenario and include random errors to parameters
sim_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/simulate_data'))
sys.path.insert(0, sim_path) 
from simulate_data.add_random_errors import add_random_errors_to_kinematic

if __name__ == '__main__':
    """
    Calibrate a heliostat using the CalibrationModel class.
    The calibration model attempts to learn the kinematic parameters of the 
    heliostat by back-propagating the offset in aimed reflection direction and
    actual reflection direction obtained from the calibration data.
    """
    
    config_file = '/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal_Sim/run_config.json'
    run_config = json.load(open(config_file))

    # Set device.
    device = torch.device(run_config['run']['device'])

    # The name of the heliostat and its index within the scenario.
    heliostat_ids = run_config['run']['heliostats']
    scenario_path = Path(run_config['run']['scenario_path'])

    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario file not found at path: {scenario_path}")
    
    # Load the scenario from the h5-file.
    with h5py.File(scenario_path, "r") as scenario_file:
        loaded_scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, 
            device=device
        )
    heliostat_ids = loaded_scenario.heliostat_field.all_heliostat_names
    
    # Define name of run which will be model name.
    run = str('run_' + datetime.now().strftime("%y%m%d%H%M") + '_' 
              + '.'.join([heliostat_id for heliostat_id in heliostat_ids]))

    # Returns a copy of the kinematic instance with changed paramaters
    modified_kinematic = add_random_errors_to_kinematic(kinematic=loaded_scenario.heliostat_field.rigid_body_kinematic, 
                                                        save_dir=Path(f'/dss/dsshome1/05/di38kid/data/results/runs/{run}') / 'parameters', 
                                                        heliostat_names=heliostat_ids,
                                                        seed=7, 
                                                        device=device)

    # Replace the 'old' kinematic with the modified kinematic
    loaded_scenario.heliostat_field.rigid_body_kinematic = modified_kinematic
    
    # Initiate Calibration Model.
    model = CalibrationModel(
        name=run,
        scenario=loaded_scenario,
        heliostat_ids=run_config['run']['heliostats'],
        calib_data_directory=run_config['run']['paint_dir'],
        config=run_config,
        random_seed=run_config['run']['random_seed'],
        device=device,
        )
    
    splits = model.calibration_data_loader.splits
    for split_type in splits.keys():
        for split_sizes in splits[split_type].keys():
            print(f"{split_type}: {split_sizes}")
            model.calibrate(num_epochs=run_config['run']['num_epochs'], 
                            log_steps=1, 
                            split_type=split_type,
                            split_sizes=split_sizes, 
                            seed=run_config['run']['random_seed'],
                            use_raytracing=bool(run_config['run']['use_raytracing']), 
                            run_config=run_config, device=device)
        
    # model.cuda()

    initial_lrs = [0.05,
                   0.0005,
                   0.5]

    param_groups = [
        ['heliostat_position_enu_4d',
         'first_joint_translation_e', 'first_joint_translation_n', 'first_joint_translation_u',
         'second_joint_translation_e', 'second_joint_translation_n', 'second_joint_translation_u',
         'concentrator_translation_e', 'concentrator_translation_n', 'concentrator_translation_u', ],
        ['first_joint_tilt_n', 'first_joint_tilt_u',
         'second_joint_tilt_e', 'second_joint_tilt_n',
         'actuator1_initial_stroke_length', 'actuator2_initial_stroke_length',
         'actuator1_offset', 'actuator2_offset',
         'actuator1_pivot_radius', 'actuator2_pivot_radius',
         'actuator1_initial_angle', 'actuator2_initial_angle'],
        ['actuator1_increment', 'actuator2_increment']
    ]

    # model.setup_optimizer(scheduler='ReduceLROnPlateau', param_groups=param_groups, initial_lr=initial_lrs)

    # exclude_parameters = \
    #     (
    #         # 'heliostat_position_enu_4d',
    #         # 'first_joint_translation_e',
    #         # 'first_joint_translation_n',
    #         # 'first_joint_translation_u',
    #         # 'second_joint_translation_e',
    #         # 'second_joint_translation_n',
    #         # 'second_joint_translation_u',
    #         # 'concentrator_translation_e',
    #         # 'concentrator_translation_n',
    #         # 'concentrator_translation_u',
    #         'first_joint_tilt_e',  # not used in kinematic model
    #         # 'first_joint_tilt_n',
    #         # 'first_joint_tilt_u',
    #         # 'second_joint_tilt_e',
    #         # 'second_joint_tilt_n',
    #         'second_joint_tilt_u',  # not used in kinematic model
    #         'concentrator_tilt_e',  # not used in kinematic model
    #         'concentrator_tilt_n',  # not used in kinematic model
    #         'concentrator_tilt_u',  # not used in kinematic model
    #         # 'actuator1_increment',
    #         # 'actuator1_initial_stroke_length',
    #         # 'actuator1_offset',
    #         # 'actuator1_pivot_radius',
    #         # 'actuator1_initial_angle',
    #         # 'actuator2_increment',
    #         # 'actuator2_initial_stroke_length',
    #         # 'actuator2_offset',
    #         # 'actuator2_pivot_radius',
    #         # 'actuator2_initial_angle'
    #    )
    
    

    # model.setup_optimizer(scheduler='ReduceLROnPlateau', param_groups=param_groups, initial_lr=initial_lrs)
    # model.model_forward(device=device)
    

    save_path = Path(r'C:\Users\wort_tr\Documents\00_Masterarbeit\04_Ergebnisse\02_Alignment\_buffer_alignment')
    # model.fit_parameters(
    #     num_epochs=150,
    #     log_steps=1,
    #     tolerance=1,  #mrad
    #     find_init_lr=False,
    #     # exclude_params=exclude_parameters,
    #     save_path=save_path,
    #     device=device)

    # model.fit_parameters_with_raytracing(num_epochs=30, log_steps=1, device=device)
