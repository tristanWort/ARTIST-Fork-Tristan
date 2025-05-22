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


def objective(params):
    threshold, final_sharpness, num_interpolations, sigma_out = params
    # Load base config
    with open("/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/image_losses/run_config.json", "r") as f:
        run_config = json.load(f)
        
    n_samples = 10
    n_epochs = 800
    tolerance = 10  # for MSE on True_Alignment
    
    # Inject hyperparams into config or pass them directly if your function allows
    # Otherwise, monkey-patch your function or define them globally in a helper module
    # mse = run_experiment(run_config, n_epochs, n_samples, threshold, sharpness, sigma_in, sigma_out, num_interpolations, tolerance)
    mse, bad_count = run_experiment(run_config, n_epochs, n_samples, threshold, final_sharpness, num_interpolations, sigma_out, tolerance)

    # Return a single float metric to minimize
    return mse  

if __name__ == '__main__':
    # Define seach space
    from skopt.space import Real, Integer
    
    search_space = [
        Real(0.05, 0.5, name="threshold"),
        # Real(0.1, 1.0, name="sharpness_sf"),  # scaling factor linearly scales from epochs -> sharpness
        Real(20, 100, name="final_sharpness"),  # scaling factor linearly scales from epochs -> sharpness
        # Real(1, 200, name="sharpness"),  # instead use some sharpness scaling factor (increase over scaling epochs)
        # Real(0.5, 5.0, name="sigma_in"),
        Integer(0, 2, name="num_interpolations"),
        Real(0.0, 10.0, name="sigma_out"),
        ]
        
    # Run Bayesian optimizer.
    res = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=100,         # You can increase this
        random_state=42,
        verbose=True,
        )
    
    print("Best parameters:", res.x)
    print("Lowest MSE on true alignment errors:", res.fun)
    
    best_params = {"Best_Parameters": [float(x) for x in res.x],
                   "Lowest_MSE": float(res.fun)}
    
    json.dump(best_params, open('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis/HeliOptiCal/image_losses/best_parameters.json', 'w+'), indent=4)
    
