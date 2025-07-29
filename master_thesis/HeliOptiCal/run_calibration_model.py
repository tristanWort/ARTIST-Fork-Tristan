import torch
import os
import sys
import h5py
import json
import gc

from datetime import datetime
from pathlib import Path

import utils.my_config_dict as my_config_dict

# Add local artist path for raytracing with multiple parallel heliostats.
artist_repo = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, artist_repo)  
from artist.util.scenario import Scenario
from calibration_model import CalibrationModel


if __name__ == '__main__':
    """
    Calibrate a heliostat using the CalibrationModel class.
    The calibration model attempts to learn the kinematic parameters of the 
    heliostat by back-propagating the offset in aimed reflection direction and
    actual reflection direction obtained from the calibration data.
    """
    
    # job_folder = Path("/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/_jobs/_jobs_0630_6")
    config_path = [Path("/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/run_config.json")]
    # blocking = 20
    
    # Iterate over all .json files in the folder
    # for json_file in job_folder.glob("*.json"):
    for json_file in config_path:
        with open(json_file, 'r', encoding='utf-8') as f:
            run_config = json.load(f)
        
        # Initiate calibration model
        model = CalibrationModel(run_config=run_config)
        model.calibrate()
        
        del model
        torch.cuda.empty_cache()  # Releases unreferenced GPU memory
        gc.collect() # Python garbage collection
    
    
    

    
    

