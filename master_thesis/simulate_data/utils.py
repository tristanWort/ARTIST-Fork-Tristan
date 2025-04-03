import torch
import os
import sys
import pathlib

from matplotlib import pyplot as plt

# Add local artist path for raytracing with multiple parallel heliostats.
repo_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, repo_path) 

from artist.raytracing.heliostat_tracing import HeliostatRayTracer


def aim_and_shoot_and_save_bitmaps(scenario,
                                   name: str,
                                   incident_ray_directions: torch.tensor,
                                   target_areas: list(),
                                   aim_points=None,
                                   align_with_motor_positions=False,
                                   motor_positions=None,
                                   seed=42,
                                   device='cuda'):
        """
        Perform aiming and raytracing for the heliostat field for the given scenario.
        
        For alignment there are two options.
            1) Align with incident ray directions and aim points.
            2) Align with motor positions.
        """
        # Auxiliary referencing to kinematic
        kinematic = scenario.heliostat_field.rigid_body_kinematic
        
        if not align_with_motor_positions:
            # Set aim points and then align with incident ray directions
            kinematic.aim_points = aim_points
            # Align with incident ray directions
            scenario.heliostat_field.align_surfaces_with_incident_ray_direction(
                incident_ray_direction=incident_ray_directions,
                round_motor_pos=True,
                device=device
                )
        
        else:
            # Align with motor positions
            print(f'Align with motor positions: {motor_positions}')
            scenario.heliostat_field.align_surfaces_with_motor_positions(
               motor_positions=motor_positions,
               device=device
               )
        
        # Initiate Raytracer
        raytracer = HeliostatRayTracer(scenario=scenario, 
                                       world_size=1, 
                                       rank=0, 
                                       batch_size=1, 
                                       random_seed=seed) 
        # Raytrace and store bitmaps
        final_bitmaps = raytracer.trace_rays_separate(incident_ray_directions=incident_ray_directions,
                                                      target_areas=target_areas,
                                                      device=device)
        
        # Save bitmaps in one img file
        fig, axs = plt.subplots(nrows=final_bitmaps.shape[0], figsize=(10, 10))
        for i in range(final_bitmaps.shape[0]):
            print(f"\tPixel sum in flux image {i+1}:", final_bitmaps[i].sum().item())
            axs[i].imshow(final_bitmaps[i].cpu().detach(), cmap="inferno")

        
        fig.suptitle(f"Flux Density Distributions: {name}")
        save_dir = pathlib.Path('/dss/dsshome1/05/di38kid/data/results/simulated_data/01/raytracing')
        save_path = save_dir / f'{name}_raytracing_sep.png'
        plt.tight_layout()
        plt.savefig(save_path)
        print("Bitmpas were saved!")
