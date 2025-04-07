import torch
import h5py
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt, animation
from torch.nn import MSELoss, L1Loss
from torchmetrics.image import StructuralSimilarityIndexMeasure
torch.autograd.set_detect_anomaly(True)

# Add local artist path for raytracing with multiple parallel heliostats.
import os
import sys
artist_repo = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/artist'))
sys.path.insert(0, artist_repo)
from artist.util import set_logger_config
from artist.util.scenario import Scenario
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.raytracing import raytracing_utils
from artist.util.paint_loader import extract_paint_calibration_data

from utils import calculate_alignment_error_mrad, calculate_image_losses, get_rigid_body_kinematic_parameters_from_scenario

set_logger_config()


if __name__ == '__main__':

    """Testing image losses for alignment for fractured flux instances."""
    
    """!!! Not working because of issues with .backward() when optimizing for aim points !!!"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the path to the working directory.
    paint_data_path = Path(r"/dss/dsshome1/05/di38kid/data/paint/")
    tower_json_file = paint_data_path / "WRI1030197-tower-measurements.json"

    # Path of the scenario which will be loaded.
    pick_scenario = '250129-1347_scenario_AA39.h5'
    scenario_path = (Path(r'/dss/dsshome1/05/di38kid/data/scenarios')
                     / pick_scenario)

    # Load the scenario.
    with h5py.File(scenario_path, "r") as f:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file=f, device=device)

    # Define the heliostat, which will be used for raytracing (scenario must exist).
    heliostat_name = "AA39"
    heliostat_index = 0
    heliostat_field = scenario.heliostat_field

    # Define the calibration ID for the heliostat.
    pick_calibration_id = '226054'  # leave empty to perform raytracing for all calibration files
    calibration_json_files = [f for f
                              in Path(f"{paint_data_path}/{heliostat_name}/Calibration").iterdir()
                              if f.name.endswith(f"{pick_calibration_id}-calibration-properties.json")]
    # Run loop for all calibration files and perform surface alignment and ray tracing.
    # print(f"\tFound {len(calibration_json_files)} calibration files.")

    calibration_json_f = calibration_json_files[0]
    (
        calibration_target_name,
        center_calibration_image,
        sun_position,
        motor_positions, 
        _
    ) = extract_paint_calibration_data(calibration_json_f, scenario.power_plant_position, device=device)

    # Get incident ray direction
    incident_ray_direction = (torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_position).unsqueeze(0)
    
    # Get target area
    target_area = [target
                   for target in scenario.target_areas.target_area_list
                   if target.name == calibration_target_name][0]

    # Raytrace with motor positions that lead to a fractured flux image
    edge_mp = (motor_positions + torch.tensor([0, 1600], device=device)).unsqueeze(0)
    heliostat_field.align_surfaces_with_motor_positions(edge_mp, device)

    # Safe orientation as it contains the surface normal and origin
    orientation = heliostat_field.rigid_body_kinematic.orientations[0]
    target_normal = orientation[0:4, 2]

    target_vector = raytracing_utils.reflect(incident_ray_direction, target_normal)

    # Define the raytracer.
    raytracer = HeliostatRayTracer(
        scenario=scenario,
        batch_size=1,
    )

    # Perform heliostat-based raytracing.
    target_image = raytracer.trace_rays_separate(
        incident_ray_directions=incident_ray_direction,
        target_areas=[target_area],
        device=device
    )[0]

    # # Plot the result.
    fig, axs = plt.subplots(3, 1, figsize=(6, 6))
    axs[0].imshow(target_image.cpu().detach().numpy(), cmap="inferno")
    axs[0].set_title(f'Target flux', pad=20)

    # Define optimizer for finding optimal aim point
    aim_points = torch.tensor([target_area.center.tolist()], requires_grad=True, device=device)
    # aim_points = aim_point.unsqueeze(0)
    heliostat_field.rigid_body_kinematic.aim_points = aim_points

    param = get_rigid_body_kinematic_parameters_from_scenario(heliostat_field.rigid_body_kinematic)
    optim = torch.optim.Adam(param.parameters(), lr=0.005)

    alignment_errors = []
    criterion_log = {'L1': [], 'MSE': [], 'CHD': [], 'HAUS': [], 'SSIM': [], 'SWD':[]}
    images = []

    best_error = torch.inf
    best_image = torch.empty_like(target_image[0])
    # tolerance = 0.5  # mrad

    for epoch in range(30):
            optim.zero_grad()

            # Use initial motor positions to set initial orientation
            # heliostat_field.align_surfaces_with_incident_ray_direction(incident_ray_direction, round_motor_pos=True, device=device)
            heliostat_field.align_surfaces_with_motor_positions(motor_positions, device=device)
            surface_normal = heliostat_field.rigid_body_kinematic.orientations[0, 0:4, 2]
            vector = raytracing_utils.reflect(incident_ray_direction, surface_normal)

            alignment_error = calculate_alignment_error_mrad(vector, target_vector)[0]
            alignment_errors.append(alignment_error.detach().item())

            # Perform heliostat-based raytracing.
            image = raytracer.trace_rays_separate(
                incident_ray_directions=incident_ray_direction,
                target_areas=[target_area],
                device=device
            )[0]
            images.append(image.cpu().detach().numpy())
            if epoch == 0:
                init_image = image

            l1, mse, chd, haus, ssim, swd = calculate_image_losses(image, target_image, device=device)
            criterion_log['L1'].append(l1.detach().item())
            criterion_log['MSE'].append(mse.detach().item())
            criterion_log['CHD'].append(chd.detach().item())
            criterion_log['HAUS'].append(haus.detach().item())
            criterion_log['SSIM'].append(ssim.detach().item())
            criterion_log['SWD'].append(swd.detach().item())

            alignment_error.backward()
            print(f'Epoch {epoch}: {alignment_error} mrad')
            optim.step()

            if alignment_error < best_error:
                best_error = alignment_error
                best_image = image

    axs[1].imshow(init_image.cpu().detach().numpy(), cmap="inferno")
    axs[1].set_title(f'Initial Flux with {alignment_errors[0]: .2f} mrad', pad=15)
    axs[2].imshow(best_image.cpu().detach().numpy(), cmap="inferno")
    axs[2].set_title(f'Best Flux with {best_error.detach().item(): .2f} mrad', pad=15)

    plt.tight_layout()
    # plt.show()
    plt.close(fig)


    # Show statistics
    fig, axs = plt.subplots(4, 2, figsize=(12, 24))
    axs[0, 0].plot(alignment_errors, label='Alignment Error')
    axs[0, 0].legend()

    im = axs[0, 1].imshow(images[0], cmap="inferno")

    def update(frame):
        im.set_array(images[frame])
        # TODO: Make animation of title work
        # axs[0, 1].set_title(f'Epoch {frame}: Flux with {alignment_errors[frame]: .2f} mrad')
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(images), blit=True)

    criteria = list(criterion_log.keys())
    j = 0
    while j < 2:
        i = 0
        while i < 3:
            criterion = criteria[i + j * 3]
            axs[i +1, j].plot(criterion_log[criterion], label=criterion)
            axs[i +1, j].legend()

            axs[i +1, j].grid()
            i += 1
        j += 1

    plt.figsave()
    # plt.show()
    ani.save('raytracing_on_image_loss.gif', writer='pillow', fps=10, dpi=100)
    plt.close(fig)

