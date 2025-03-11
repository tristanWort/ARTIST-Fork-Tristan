import torch
import gpustat
import psutil
from artist.field.kinematic_rigid_body import RigidBody

def get_rigid_body_kinematic_parameters_from_scenario(
    kinematic: RigidBody,
) -> dict[str, torch.Tensor]:
    """
    Extract all deviation parameters and actuator parameters from a rigid body kinematic.

    Parameters
    ----------
    kinematic : RigidBody
        The kinematic from which to extract the parameters.

    Returns
    -------
    dict[str, torch.Tensor]
        The parameters from the kinematic (requires_grad is True).
    """

    parameters_dict = {
        "heliostat_position_enu_4d": kinematic.position,
        "first_joint_translation_e": kinematic.deviation_parameters.first_joint_translation_e,
        "first_joint_translation_n": kinematic.deviation_parameters.first_joint_translation_n,
        "first_joint_translation_u": kinematic.deviation_parameters.first_joint_translation_u,
        "first_joint_tilt_e": kinematic.deviation_parameters.first_joint_tilt_e,
        "first_joint_tilt_n": kinematic.deviation_parameters.first_joint_tilt_n,
        "first_joint_tilt_u": kinematic.deviation_parameters.first_joint_tilt_u,
        "second_joint_translation_e": kinematic.deviation_parameters.second_joint_translation_e,
        "second_joint_translation_n": kinematic.deviation_parameters.second_joint_translation_n,
        "second_joint_translation_u": kinematic.deviation_parameters.second_joint_translation_u,
        "second_joint_tilt_e": kinematic.deviation_parameters.second_joint_tilt_e,
        "second_joint_tilt_n": kinematic.deviation_parameters.second_joint_tilt_n,
        "second_joint_tilt_u": kinematic.deviation_parameters.second_joint_tilt_u,
        "concentrator_translation_e": kinematic.deviation_parameters.concentrator_translation_e,
        "concentrator_translation_n": kinematic.deviation_parameters.concentrator_translation_n,
        "concentrator_translation_u": kinematic.deviation_parameters.concentrator_translation_u,
        "concentrator_tilt_e": kinematic.deviation_parameters.concentrator_tilt_e,
        "concentrator_tilt_n": kinematic.deviation_parameters.concentrator_tilt_n,
        "concentrator_tilt_u": kinematic.deviation_parameters.concentrator_tilt_u,
        "actuator1_increment": kinematic.actuators.actuator_list[0].increment,
        "actuator1_initial_stroke_length": kinematic.actuators.actuator_list[0].initial_stroke_length,
        "actuator1_offset": kinematic.actuators.actuator_list[0].offset,
        "actuator1_pivot_radius": kinematic.actuators.actuator_list[0].pivot_radius,
        "actuator1_initial_angle": kinematic.actuators.actuator_list[0].initial_angle,
        "actuator2_increment": kinematic.actuators.actuator_list[1].increment,
        "actuator2_initial_stroke_length": kinematic.actuators.actuator_list[1].initial_stroke_length,
        "actuator2_offset": kinematic.actuators.actuator_list[1].offset,
        "actuator2_pivot_radius": kinematic.actuators.actuator_list[1].pivot_radius,
        "actuator2_initial_angle": kinematic.actuators.actuator_list[1].initial_angle,
    }

    for parameter in parameters_dict.values():
        if parameter is not None:
            parameter.requires_grad_()

    return parameters_dict

def enu_point_to_target_plane(
        enu_point: torch.Tensor,
        target_center: torch.Tensor,
        plane_e: float,
        plane_u: float,
        resolution_e: int=256,
        resolution_u: int=256,
) -> list:
    """
    Map the coordinates of a point in ENU coordinates onto a target plane.

    Parameters
    ----------
    enu_point
    target_center
    plane_e
    plane_u

    Returns
    -------

    """

    d_enu = enu_point - target_center
    map_e = 0.5 + d_enu[0].item() / plane_e
    map_u = 0.5 + d_enu[2].item() / plane_u
    pixel_point = [
        resolution_e * map_e,
        resolution_u * map_u,
    ]
    return pixel_point

def percentage_near_max(image, threshold=1):
    """
    Calculate the percentage of pixels within a certain threshold of the maximum value
    in each image of the (n, w, h) tensor.


    Args:
        image (torch.Tensor):
            Input-Tensor of shape (n, w, h).
        threshold (float):
            Distance from the maximum value within which pixels are counted.

    Returns:
        numpy.ndarray oder torch.Tensor:
            An array of percentage values for each image in the batch.
    """
    if isinstance(image, torch.Tensor):
        max_vals = image.amax(dim=(1, 2), keepdim=True)  # Maximalwerte pro Bild
        mask = (image >= (max_vals - threshold))  # Maske für Pixel innerhalb des Thresholds
        percentages = mask.sum(dim=(1, 2)) / (image.shape[1] * image.shape[2]) * 100
        return percentages

    else:
        raise TypeError("Input must be a PyTorch-Tensor!")

def print_gpu_memory_usage(device, location=''):
    """Monitor GPU memory usage"""
    print("-------------------------------------------------------------------------")
    print(f"Location: {location}")
    print(f"CPU memory usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")
    mem = psutil.virtual_memory()
    total_ram = mem.total / (1024 ** 3)  # Convert bytes to GB
    print(f"Available RAM on CPU: {total_ram:.2f} GB")
    print(f"Allocated memory: {torch.cuda.memory_allocated(device=device) / 1024**2:.2f} MB")
    print(f"Reserved memory: {torch.cuda.memory_reserved(device=device) / 1024**2:.2f} MB")
    print(f"Max allocated memory: {torch.cuda.max_memory_allocated(device=device) / 1024**2:.2f} MB")
    print(f"Max reserved memory: {torch.cuda.max_memory_reserved(device=device) / 1024**2:.2f} MB")
    print(gpustat.new_query())
    print("-------------------------------------------------------------------------")
