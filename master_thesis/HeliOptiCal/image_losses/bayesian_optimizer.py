"""
Applying Bayesian optimization for finding optimal parameters for contour based alignment.
"""
from skopt import gp_minimize
import json
import os
import sys

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal'))
sys.path.insert(0, repo_path)
from HeliOptiCal.image_losses.bayesian_experiment import run_experiment

# h_idx = 0

def objective(params):
    threshold, sharpness, sigma_in, sigma_out, num_interpolations, beta, gamma, epsilon1, epsilon2, omega = params
    # Load base config
    with open("/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/image_losses/run_config.json", "r") as f:
        run_config = json.load(f)
        
    n_samples = 10
    n_epochs = 800
    tolerance = 10  # for MSE on True_Alignment

    # Inject hyperparams into config or pass them directly if your function allows
    mse = run_experiment(run_config, n_epochs, n_samples, 
                         threshold, sharpness, sigma_in, sigma_out, num_interpolations, beta, gamma, epsilon1, epsilon2, omega,
                         tolerance)

    # Return a single float metric to minimize
    return mse  


if __name__ == '__main__':
    # Define seach space
    from skopt.space import Real, Integer
    
    search_space = [
        Real(0.1, 0.9, name="threshold"),
        # Real(0.1, 1.0, name="sharpness_sf"),  # scaling factor linearly scales from epochs -> sharpness
        # Real(20, 100, name="final_sharpness"),  # scaling factor linearly scales from epochs -> sharpness
        Integer(20, 80, name="sharpness"),  # instead use some sharpness scaling factor (increase over scaling epochs)
        Real(10.0, 60.0, name="sigma_in"),
        Real(5.0, 50.0, name="sigma_out"),
        Integer(3, 6, name="num_interpolations"),
        Real(0.3, 0.8, name="beta"),
        Real(0.05, 0.3, name="gamma"),
        Real(0.1, 0.9, name="epsilon1"),
        Real(0.1, 0.9, name="epsilon2"),
        Real(0.1, 0.9, name="omega"),
        # Integer(0, 1, name='use_dist'),
        # Real(0.1, 1.0, name='dist_factor')
        ]
    
    # Run Bayesian optimizer.
    res = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=35,         # You can increase this
        random_state=42,
        verbose=True,
        )
    
    print("Best parameters:", res.x)
    print("Lowest MSE on true alignment errors:", res.fun)
    
    best_params = {"Best_Parameters": [float(x) for x in res.x],
                "Lowest_MSE": float(res.fun)}
    
    json.dump(best_params, open(f'/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/image_losses/best_parameters.json', 'w+'), indent=4)
    
