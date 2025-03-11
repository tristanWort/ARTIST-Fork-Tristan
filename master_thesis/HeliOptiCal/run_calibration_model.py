import torch
import os
import h5py

from datetime import datetime
from pathlib import Path

from artist.scenario import Scenario
from calibration_model import CalibrationModel

from artist.util.configuration_classes import (
    LightSourceConfig, LightSourceListConfig)

if __name__ == '__main__':
    """Calibrate a heliostat using the CalibrationModel class.
    The calibration model attempts to learn the kinematic parameters of the 
    heliostat by back-propagating the offset in aimed reflection direction and
    actual reflection direction obtained from the calibration data."""
    # torch.set_printoptions(precision=10)
    # Set device to GPU if available, else to CPU.
    # device = torch.device("cpu")
    # print(device)
    device = torch.device('cpu')
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    # The name of the heliostat and its index within the scenario.
    time_stamp = datetime.now().strftime("%y%m%d%H%M")
    heliostat = 'AM35'  # will be used to find the scenario file
    scenario_path = Path(r'/jump/tw/data/paint/_h5_scenario_files/250206-1455_scenario_AM35.h5')

    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario file not found at path: "
                                f"{scenario_path}")

    # Load the scenario from the h5-file.
    print("Load Scenario...")
    with h5py.File(scenario_path, "r") as scenario_file:
        one_scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    one_scenario.light_source_list_config = LightSourceListConfig(
        [LightSourceConfig(
            light_source_key="sun1",
            light_source_type="sun",
            number_of_rays=4,
            distribution_type="normal",
            mean=0.0,
            covariance=2*4.3681e-06  # for more realistic raytracing
        )]
    )

    # Give the path to the folder containing the calibration data.
    print("Load calibration data...")
    one_calibration_data_path = (Path(r'/jump/tw/data/paint/') /
                                 f'{heliostat}/Calibration')
    
    if not os.path.exists(one_calibration_data_path):
        raise FileNotFoundError(f"Calibration folder not found at path: "
                                f"{one_calibration_data_path}")

    test_raytracing_for_these_calibrations = [
        '135907', '137687', '155687', '172123', '190707',
        '90739', '91533', '126109', '128286', '131204',
       '65720', '242059', '241624', '217062', '192343'
    ]

    print("Build model...")
    model_name = time_stamp + '_' + heliostat
    model = CalibrationModel(
        name=model_name,
        scenario=one_scenario,
        calibration_data_path=one_calibration_data_path,
        heliostat_index=0,
        # calibration_ids=test_raytracing_for_these_calibrations,
        device=device)

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

    model.setup_optimizer(scheduler='ReduceLROnPlateau', param_groups=param_groups, initial_lr=initial_lrs)

    exclude_parameters = \
        (
            # 'heliostat_position_enu_4d',
            # 'first_joint_translation_e',
            # 'first_joint_translation_n',
            # 'first_joint_translation_u',
            # 'second_joint_translation_e',
            # 'second_joint_translation_n',
            # 'second_joint_translation_u',
            # 'concentrator_translation_e',
            # 'concentrator_translation_n',
            # 'concentrator_translation_u',
            'first_joint_tilt_e',  # not used in kinematic model
            # 'first_joint_tilt_n',
            # 'first_joint_tilt_u',
            # 'second_joint_tilt_e',
            # 'second_joint_tilt_n',
            'second_joint_tilt_u',  # not used in kinematic model
            'concentrator_tilt_e',  # not used in kinematic model
            'concentrator_tilt_n',  # not used in kinematic model
            'concentrator_tilt_u',  # not used in kinematic model
            # 'actuator1_increment',
            # 'actuator1_initial_stroke_length',
            # 'actuator1_offset',
            # 'actuator1_pivot_radius',
            # 'actuator1_initial_angle',
            # 'actuator2_increment',
            # 'actuator2_initial_stroke_length',
            # 'actuator2_offset',
            # 'actuator2_pivot_radius',
            # 'actuator2_initial_angle'
        )
    
    model.calibrate(num_epochs=30, log_steps=1, train_valid_test_sizes=(30,15,15), exclude_params=exclude_parameters, with_raytracing=True, device=device)

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
