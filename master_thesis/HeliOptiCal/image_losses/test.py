from skopt import gp_minimize
import json
import os
import sys

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, repo_path)
from HeliOptiCal.image_losses.bayesian_experiment import run_experiment

h_idx = 0

threshold = 0.3
sharpness = 60
sigma_in = 1.5
sigma_out = 1.5
num_interpolations = 2
beta = 0.0

# Load base config
with open("/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/image_losses/run_config.json", "r") as f:
    run_config = json.load(f)
    
n_samples = 10
n_epochs = 800
tolerance = 2  # for MSE on True_Alignment

# Inject hyperparams into config or pass them directly if your function allows
mse = run_experiment(run_config, n_epochs, n_samples, h_idx, threshold, sharpness, sigma_in, sigma_out, num_interpolations, tolerance)
