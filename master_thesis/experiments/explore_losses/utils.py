import torch
from torch.nn import MSELoss, L1Loss
from torchmetrics.image import StructuralSimilarityIndexMeasure


def calculate_alignment_error_mrad(v1, v2, epsilon: float = 1e-8):
    m1 = torch.norm(v1, dim=-1)
    m2 = torch.norm(v2, dim=-1)
    dot_products = torch.sum(v1 * v2, dim=-1)
    cos_angles = dot_products / (m1 * m2 + epsilon)
    angles_rad = torch.acos(
        torch.clamp(cos_angles, min=-1.0 + 1e-7, max=1.0 - 1e-7)
    )

    return angles_rad * 1000


# Add local path to modify scenario and include random errors to parameters
import sys
import os
sim_path = os.path.abspath(os.path.dirname('/dss/dsshome1/05/di38kid/master_thesis/ARTIST-Fork-Tristan/master_thesis'))
sys.path.insert(0, sim_path)
from master_thesis.HeliOptiCal.image_loss import chamfer_distance_batch, hausdorff_loss, SlicedWassersteinDistance


def calculate_image_losses(image1, image2, device='cuda'):
    l1_func = L1Loss().to(device)
    l1 = l1_func(image1, image2)

    mse_func = MSELoss().to(device)
    mse = mse_func(image1, image2)

    chd = chamfer_distance_batch(
        image1.unsqueeze(0),
        image2.unsqueeze(0),
    ).mean()

    haus = hausdorff_loss(
        image1.unsqueeze(0),
        image2.unsqueeze(0),
    )

    ssim_func = StructuralSimilarityIndexMeasure().to(device)
    ssim = ssim_func(image1.unsqueeze(0).unsqueeze(0), image2.unsqueeze(0).unsqueeze(0))

    swd_func = SlicedWassersteinDistance().to(device)
    swd = swd_func(image1.unsqueeze(0).unsqueeze(0), image2.unsqueeze(0).unsqueeze(0))

    return l1, mse, chd, haus, ssim, swd


def get_rigid_body_kinematic_parameters_from_scenario(
    kinematic: "RigidBody",
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
    parameters_dict = torch.nn.ParameterDict()
    parameters_dict.update(kinematic.all_heliostats_position_params)
    # parameters_dict.update(kinematic.all_deviations_params)
    # parameters_dict.update(kinematic.all_actuators_params)
    
    return parameters_dict

