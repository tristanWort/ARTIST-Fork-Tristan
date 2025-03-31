from typing import Union

from artist.field.actuator_linear import LinearActuators
import torch

from artist.field.kinematic import (
    Kinematic,
)
from artist.util import config_dictionary, utils


class RigidBody(Kinematic):
    """
    Implement a rigid body kinematic model.

    Attributes
    ----------
    number_of_heliostats : int
        The number of heliostats using a rigid body kinematic.
    heliostat_positions : torch.Tensor
        The positions of the heliostats.
    aim_points : torch.Tensor
        The aim points of the heliostats.
    initial_orientations : torch.Tensor
        The initial orientation offsets of the heliostats.
    deviation_parameters : torch.Tensor
        The deviation parameters for the kinematic.
    artist_standard_orientation : torch.Tensor
        The standard orientation of the kinematic.
    actuators : LinearActuators
        The linear actuators of the kinematic.

    Methods
    -------
    incident_ray_direction_to_orientation()
        Compute orientation matrices given an incident ray direction.
    align_surface_with_incident_ray_direction()
        Align given surface points and surface normals according to an incident ray direction.
    motor_positions_to_orientation()
        Compute the orientation matrix given the motor positions.
    align_surface_with_motor_positions()
        Align given surface points and surface normals according to motor positions.
    forward()
        Specify the forward pass.

    See Also
    --------
    :class:`Kinematic` : Reference to the parent class.
    """
    def __init__(
        self,
        number_of_heliostats: int,
        heliostat_positions: torch.Tensor,
        aim_points: torch.Tensor,
        actuator_parameters: torch.Tensor,
        initial_orientations: torch.Tensor,
        deviation_parameters: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """
        Initialize the rigid body kinematic.

        The rigid body kinematic determines a transformation matrix that is applied to the heliostat surfaces in order to
        align them. The heliostats then reflect the incoming light according to the provided aim points. The rigid body 
        kinematic works for heliostats equipped with two actuators that turn the heliostat surfaces.
        Furthermore, initial orientation offsets and deviation parameters determine the specific behavior of the kinematic.

        Parameters
        ----------
        number_of_heliostats : int
            The number of heliostats using a rigid body kinematic.
        all_heliostats_position_params : torch.nn.ParameterDict
            The positional parameters of the heliostats.
        aim_points : torch.Tensor
            The aim points of the heliostats.
        all_actuators_params : torch.nn.ParameterDict
            The actuator parameters.
        all_actuators : LinearActuators.
            All linear actuators as one LinearActuators instance.
        initial_orientations : torch.Tensor
            The initial orientation offsets of the heliostats.
        all_deviations_params : torch.nn.ParameterDict
            The deviation parameters for the kinematic.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).
        """
        super().__init__()
        device = torch.device(device)

        self.number_of_heliostats = number_of_heliostats
        self.aim_points = aim_points
        self.initial_orientations = initial_orientations
        self.orientations = initial_orientations
        
        self.all_heliostats_position_params, self.all_deviations_params = self._process_parameters(
            all_heliostat_positions=heliostat_positions,
            all_deviations=deviation_parameters,
            device=device
            )
        
        self.all_actuators_params, self.actuators = self._config_actuators(
            all_actuator_parameters=actuator_parameters, 
            device=device
            )
        
        self.artist_standard_orientation = torch.tensor(
            [0.0, -1.0, 0.0, 0.0], device=device
        )

    def _process_parameters(
        self, 
        all_heliostat_positions: torch.Tensor,
        all_deviations: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> (torch.nn.ParameterDict, torch.nn.ParameterDict):
        device = torch.device(device)
        
        assert all_heliostat_positions.shape[0] == self.number_of_heliostats, (
            "First shape in heliostat_positions does not match the number of heliostats.")
        assert all_deviations.shape[0] == self.number_of_heliostats, (
            "First shape in deviation_parameters does not match the number of heliostats.")
        
        # Create nested paramater lists.  
        all_heliostats_position_params = torch.nn.ParameterDict({
            'heliostat_e': torch.nn.ParameterList(),   
            'heliostat_n': torch.nn.ParameterList(), 
            'heliostat_u': torch.nn.ParameterList(),
        })
        
        all_deviations_params = torch.nn.ParameterDict({
            'first_joint_translation_e': torch.nn.ParameterList(),
            'first_joint_translation_n': torch.nn.ParameterList(),
            'first_joint_translation_u': torch.nn.ParameterList(),
            # 'first_joint_tilt_e': torch.nn.ParameterList(),  # not used
            'first_joint_tilt_n': torch.nn.ParameterList(),
            'first_joint_tilt_u': torch.nn.ParameterList(),
            'second_joint_translation_e': torch.nn.ParameterList(),
            'second_joint_translation_n': torch.nn.ParameterList(),
            'second_joint_translation_u': torch.nn.ParameterList(),
            'second_joint_tilt_e': torch.nn.ParameterList(),
            'second_joint_tilt_n': torch.nn.ParameterList(),
            # 'second_joint_tilt_u': torch.nn.ParameterList(),  # not used
            'concentrator_translation_e': torch.nn.ParameterList(),
            'concentrator_translation_n': torch.nn.ParameterList(),
            'concentrator_translation_u': torch.nn.ParameterList(),
            'concentrator_tilt_e': torch.nn.ParameterList(),
            'concentrator_tilt_n': torch.nn.ParameterList(),
            'concentrator_tilt_u': torch.nn.ParameterList(),
        })
        
        # Iterate over heliostats and append the parameters.
        for heliostat_position in all_heliostat_positions:
            all_heliostats_position_params['heliostat_e'].append(torch.nn.Parameter(heliostat_position[0].to(device)))
            all_heliostats_position_params['heliostat_n'].append(torch.nn.Parameter(heliostat_position[1].to(device)))
            all_heliostats_position_params['heliostat_u'].append(torch.nn.Parameter(heliostat_position[2].to(device)))
            
        for heliostat_deviations in all_deviations:
            all_deviations_params['first_joint_translation_e'].append(torch.nn.Parameter(heliostat_deviations[0].to(device)))
            all_deviations_params['first_joint_translation_n'].append(torch.nn.Parameter(heliostat_deviations[1].to(device)))
            all_deviations_params['first_joint_translation_u'].append(torch.nn.Parameter(heliostat_deviations[2].to(device)))
            all_deviations_params['first_joint_tilt_n'].append(torch.nn.Parameter(heliostat_deviations[4].to(device)))
            all_deviations_params['first_joint_tilt_u'].append(torch.nn.Parameter(heliostat_deviations[5].to(device)))
            all_deviations_params['second_joint_translation_e'].append(torch.nn.Parameter(heliostat_deviations[6].to(device)))
            all_deviations_params['second_joint_translation_n'].append(torch.nn.Parameter(heliostat_deviations[7].to(device)))
            all_deviations_params['second_joint_translation_u'].append(torch.nn.Parameter(heliostat_deviations[8].to(device)))
            all_deviations_params['second_joint_tilt_e'].append(torch.nn.Parameter(heliostat_deviations[9].to(device)))
            all_deviations_params['second_joint_tilt_n'].append(torch.nn.Parameter(heliostat_deviations[10].to(device)))
            all_deviations_params['concentrator_translation_e'].append(torch.nn.Parameter(heliostat_deviations[12].to(device)))
            all_deviations_params['concentrator_translation_n'].append(torch.nn.Parameter(heliostat_deviations[13].to(device)))
            all_deviations_params['concentrator_translation_u'].append(torch.nn.Parameter(heliostat_deviations[14].to(device)))
            all_deviations_params['concentrator_tilt_e'].append(torch.nn.Parameter(heliostat_deviations[15].to(device)))
            all_deviations_params['concentrator_tilt_n'].append(torch.nn.Parameter(heliostat_deviations[16].to(device)))
            all_deviations_params['concentrator_tilt_u'].append(torch.nn.Parameter(heliostat_deviations[17].to(device)))
        
        return all_heliostats_position_params, all_deviations_params
        
    
    def _config_actuators(
        self, 
        all_actuator_parameters: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
        ) -> (torch.nn.ParameterDict, LinearActuators):
        device = torch.device(device)
        
        assert all_actuator_parameters.shape[0] == self.number_of_heliostats, (
            "First shape in actuatotor_parameters does not match the number of heliostats.")
        
        if (all_actuator_parameters[:, 0] != 0.0).any():
            raise NotImplementedError("Actuator type not implemented!")

        all_actuators_params = torch.nn.ParameterDict({
            'increments': torch.nn.ParameterList(),
            'initial_stroke_lengths': torch.nn.ParameterList(),
            'offsets': torch.nn.ParameterList(),
            'pivot_radii': torch.nn.ParameterList(),
            'initial_angles': torch.nn.ParameterList(),
        })
        
        for heliostat_actuators in all_actuator_parameters:
            all_actuators_params['increments'].append(torch.nn.ParameterList(
                [torch.nn.Parameter(actuator_increment).to(device) for actuator_increment in heliostat_actuators[2]]
                ))
            all_actuators_params['initial_stroke_lengths'].append(torch.nn.ParameterList(
                [torch.nn.Parameter(actuator_increment).to(device) for actuator_increment in heliostat_actuators[3]]
                ))
            all_actuators_params['offsets'].append(torch.nn.ParameterList(
                [torch.nn.Parameter(actuator_increment).to(device) for actuator_increment in heliostat_actuators[4]]
                ))
            all_actuators_params['pivot_radii'].append(torch.nn.ParameterList(
                [torch.nn.Parameter(actuator_increment).to(device) for actuator_increment in heliostat_actuators[5]]
                ))
            all_actuators_params['initial_angles'].append(torch.nn.ParameterList(
                [torch.nn.Parameter(actuator_increment).to(device) for actuator_increment in heliostat_actuators[6]]
                ))
        
        # Create one Linear Actuators instance for all heliostats and all actuators.
        all_actuators = LinearActuators(
            clockwise_axis_movements=all_actuator_parameters[:, 1],
            increments=all_actuators_params['increments'],
            initial_stroke_lengths=all_actuators_params['initial_stroke_lengths'],
            offsets=all_actuators_params['offsets'],
            pivot_radii=all_actuators_params['pivot_radii'],
        	initial_angles=all_actuators_params['initial_angles'],
        )
        
        return all_actuators_params, all_actuators
        
    def incident_ray_direction_to_orientation(
        self,
        incident_ray_direction: torch.Tensor,
        max_num_iterations: int = 2,
        min_eps: float = 0.0001,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Compute orientation matrices given an incident ray direction.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The direction of the incident ray as seen from the heliostat.
        max_num_iterations : int
            Maximum number of iterations (default: 2).
        min_eps : float
            Convergence criterion (default: 0.0001).
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The orientation matrix.
        """
        device = torch.device(device)
        
        motor_positions = torch.zeros((self.number_of_heliostats, config_dictionary.rigid_body_number_of_actuators), device=device)
        last_iteration_loss = None
        
        for _ in range(max_num_iterations):
            joint_angles = self.actuators.motor_positions_to_angles(
                motor_positions=motor_positions, device=device
            )
            
            initial_orientations = torch.eye(4, device=device).unsqueeze(0)
            
            # Account for position.
            initial_orientations = initial_orientations @ utils.translate_enu(
                e=torch.stack([p for p in self.all_heliostats_position_params['heliostat_e']]).to(device),
                n=torch.stack([p for p in self.all_heliostats_position_params['heliostat_n']]).to(device),
                u=torch.stack([p for p in self.all_heliostats_position_params['heliostat_u']]).to(device),
                device=device,
            )

            joint_rotations = torch.zeros((self.number_of_heliostats, config_dictionary.rigid_body_number_of_actuators, 4, 4), device=device)

            joint_rotations[:, 0] = (
                utils.rotate_n(
                    n=torch.stack([p for p in self.all_deviations_params['first_joint_tilt_n']]).to(device), device=device
                )
                @ utils.rotate_u(
                    u=torch.stack([p for p in self.all_deviations_params['first_joint_tilt_u']]).to(device), device=device
                )
                @ utils.translate_enu(
                    e=torch.stack([p for p in self.all_deviations_params['first_joint_translation_e']]).to(device),
                    n=torch.stack([p for p in self.all_deviations_params['first_joint_translation_n']]).to(device),
                    u=torch.stack([p for p in self.all_deviations_params['first_joint_translation_u']]).to(device),
                    device=device,)
                
                @ utils.rotate_e(
                    e=joint_angles[:, 0], device=device)
            )

            joint_rotations[:, 1] = (
                utils.rotate_e(
                    e=torch.stack([p for p in self.all_deviations_params['second_joint_tilt_e']]).to(device), device=device
                )
                @ utils.rotate_n(
                    n=torch.stack([p for p in self.all_deviations_params['second_joint_tilt_n']]).to(device), device=device
                )
                @ utils.translate_enu(
                    e=torch.stack([p for p in self.all_deviations_params['second_joint_translation_e']]).to(device),
                    n=torch.stack([p for p in self.all_deviations_params['second_joint_translation_n']]).to(device),
                    u=torch.stack([p for p in self.all_deviations_params['second_joint_translation_u']]).to(device),
                    device=device,
                )
                @ utils.rotate_u(
                    u=joint_angles[:, 1], device=device)
            )

            orientations = (
                initial_orientations
                @ joint_rotations[:, 0]
                @ joint_rotations[:, 1]
                @ utils.translate_enu(
                    e=torch.stack([p for p in self.all_deviations_params['concentrator_translation_e']]).to(device),
                    n=torch.stack([p for p in self.all_deviations_params['concentrator_translation_n']]).to(device),
                    u=torch.stack([p for p in self.all_deviations_params['concentrator_translation_u']]).to(device),
                    device=device,
                )
                @ utils.rotate_e(
                    e=torch.stack([p for p in self.all_deviations_params['concentrator_tilt_e']]).to(device),
                    device=device
                )
                @ utils.rotate_n(
                    n=torch.stack([p for p in self.all_deviations_params['concentrator_tilt_n']]).to(device),
                    device=device
                )
                @ utils.rotate_u(
                    u=torch.stack([p for p in self.all_deviations_params['concentrator_tilt_u']]).to(device),
                    device=device
                )
            )

            concentrator_normals = orientations @ torch.tensor(
                [0, -1, 0, 0], dtype=torch.float32, device=device
            )
            concentrator_origins = orientations @ torch.tensor(
                [0, 0, 0, 1], dtype=torch.float32, device=device
            )

            # Compute desired normals.
            desired_reflect_vecs = torch.nn.functional.normalize(self.aim_points - concentrator_origins, p=2, dim=1)
            desired_concentrator_normals = torch.nn.functional.normalize(-incident_ray_direction + desired_reflect_vecs, p=2, dim=1)

            # Compute epoch loss.
            loss = torch.abs(desired_concentrator_normals - concentrator_normals).mean(dim=0)

            # Stop if converged.
            if isinstance(last_iteration_loss, torch.Tensor):
                eps = torch.abs(last_iteration_loss - loss)
                if torch.any(eps <= min_eps):
                    break
            last_iteration_loss = loss

            # Analytical Solution
            joint_angles = torch.zeros((self.number_of_heliostats, config_dictionary.rigid_body_number_of_actuators), device=device)

            # Calculate joint 2 angles.
            joint_angles[:, 1] = -torch.arcsin(
                -desired_concentrator_normals[:, 0]
                / torch.cos(torch.stack([p for p in self.all_deviations_params['second_joint_translation_n']]).to(device))
            )

            # Calculate joint 1 angles.
            a = -torch.cos(
                torch.stack([p for p in self.all_deviations_params['second_joint_translation_e']]).to(device)
            ) * torch.cos(joint_angles[:, 1]) + torch.sin(
                torch.stack([p for p in self.all_deviations_params['second_joint_translation_e']]).to(device)
            ) * torch.sin(
                torch.stack([p for p in self.all_deviations_params['second_joint_translation_n']]).to(device)
            ) * torch.sin(joint_angles[:, 1])
            b = -torch.sin(
                torch.stack([p for p in self.all_deviations_params['second_joint_translation_e']]).to(device)
            ) * torch.cos(joint_angles[:, 1]) - torch.cos(
                torch.stack([p for p in self.all_deviations_params['second_joint_translation_e']]).to(device)
            ) * torch.sin(
                torch.stack([p for p in self.all_deviations_params['second_joint_translation_n']]).to(device)
            ) * torch.sin(joint_angles[:, 1])

            joint_angles[:, 0] = (
                torch.arctan2(
                    a * -desired_concentrator_normals[:, 2]
                    - b * -desired_concentrator_normals[:, 1],
                    a * -desired_concentrator_normals[:, 1]
                    + b * -desired_concentrator_normals[:, 2],
                )
                - torch.pi
            )

            motor_positions = self.actuators.angles_to_motor_positions(
                joint_angles, device
            )

        east_angles, north_angles, up_angles = utils.decompose_rotations(
            initial_vector=self.initial_orientations[:, :-1],
            target_vector=self.artist_standard_orientation[:-1],
            device=device,
        )
        # Return orientation matrices multiplied by the initial orientation offsets.
        return (
            orientations
            @ utils.rotate_e(
                e=east_angles,
                device=device,
            )
            @ utils.rotate_n(
                n=north_angles,
                device=device,
            )
            @ utils.rotate_u(
                u=up_angles,
                device=device,
            )
        )

    def align_surfaces_with_incident_ray_direction(
        self,
        incident_ray_direction: torch.Tensor,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Align given surface points and surface normals according to an incident ray direction.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The direction of the rays.
        surface_points : torch.Tensor
            Points on the surface of the heliostats that reflect the light.
        surface_normals : torch.Tensor
            Normals to the surface points.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The aligned surface points.
        torch.Tensor
            The aligned surface normals.
        """
        device = torch.device(device)

        orientations = self.incident_ray_direction_to_orientation(
            incident_ray_direction,
            max_num_iterations=10, device=device
        )
        self.orientations = orientations

        aligned_surface_points = surface_points @ orientations.transpose(1, 2)
        aligned_surface_normals = surface_normals @ orientations.transpose(1, 2)

        return aligned_surface_points, aligned_surface_normals

    def motor_positions_to_orientations(
        self,
        motor_positions: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """
        Compute the orientation matrix given the motor positions.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions from the calibration.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The orientation matrix.
        """
        # if len(self.actuators.actuator_list) != 2:
        #     raise ValueError(
        #         f"The rigid body kinematic requires exactly two actuators but {len(self.actuators.actuator_list)} were specified, please check the configuration!"
        #     )

        device = torch.device(device)
        
        joint_angles = self.actuators.motor_positions_to_angles(
                motor_positions=motor_positions, device=device
            )

        initial_orientations = torch.eye(4, device=device).unsqueeze(0)
        
        # Account for position.
        initial_orientations = initial_orientations @ utils.translate_enu(
            e=torch.stack([p for p in self.all_heliostats_position_params['heliostat_e']]).to(device),
            n=torch.stack([p for p in self.all_heliostats_position_params['heliostat_n']]).to(device),
            u=torch.stack([p for p in self.all_heliostats_position_params['heliostat_u']]).to(device),
            device=device,
        )

        joint_rotations = torch.zeros((self.number_of_heliostats, config_dictionary.rigid_body_number_of_actuators, 4, 4), device=device) 

        joint_rotations[:, 0] = (
                utils.rotate_n(
                    n=torch.stack([p for p in self.all_deviations_params['first_joint_tilt_n']]).to(device), device=device
                )
                @ utils.rotate_u(
                    u=torch.stack([p for p in self.all_deviations_params['first_joint_tilt_u']]).to(device), device=device
                )
                @ utils.translate_enu(  
                    e=torch.stack([p for p in self.all_deviations_params['first_joint_translation_e']]).to(device),
                    n=torch.stack([p for p in self.all_deviations_params['first_joint_translation_n']]).to(device),
                    u=torch.stack([p for p in self.all_deviations_params['first_joint_translation_u']]).to(device),
                    device=device,
                )
                @ utils.rotate_e(
                    e=joint_angles[:, 0], device=device)  # first_joint_angle
            )
        joint_rotations[:, 1] = (
            utils.rotate_e(
                e=torch.stack([p for p in self.all_deviations_params['second_joint_tilt_e']]).to(device), device=device
            )
            @ utils.rotate_n(
                n=torch.stack([p for p in self.all_deviations_params['second_joint_tilt_n']]).to(device), device=device
            )
            @ utils.translate_enu(
                e=torch.stack([p for p in self.all_deviations_params['second_joint_translation_e']]).to(device),
                n=torch.stack([p for p in self.all_deviations_params['second_joint_translation_n']]).to(device),
                u=torch.stack([p for p in self.all_deviations_params['second_joint_translation_u']]).to(device),
                device=device,
            )
            @ utils.rotate_u(
                u=joint_angles[:, 1], device=device)
        )

        orientations = (
            initial_orientations
            @ joint_rotations[:, 0]  
            @ joint_rotations[:, 1]
            @ utils.translate_enu(
                e=torch.stack([p for p in self.all_deviations_params['concentrator_translation_e']]).to(device),
                n=torch.stack([p for p in self.all_deviations_params['concentrator_translation_n']]).to(device),
                u=torch.stack([p for p in self.all_deviations_params['concentrator_translation_u']]).to(device),
                device=device,
            )
            @ utils.rotate_e(
                e=torch.stack([p for p in self.all_deviations_params['concentrator_tilt_e']]).to(device),
                device=device
            )
            @ utils.rotate_n(
                n=torch.stack([p for p in self.all_deviations_params['concentrator_tilt_n']]).to(device),
                device=device
            )
            @ utils.rotate_u(
                u=torch.stack([p for p in self.all_deviations_params['concentrator_tilt_u']]).to(device),
                device=device
            )
        )

        east_angles, north_angles, up_angles = utils.decompose_rotations(
            initial_vector=self.initial_orientations[:, :-1],
            target_vector=self.artist_standard_orientation[:-1],
            device=device,
        )

        # Return orientation matrix multiplied by the initial orientation offset.
        return (
            orientations
            @ utils.rotate_e(
                e=east_angles,
                device=device,
            )
            @ utils.rotate_n(
                n=north_angles,
                device=device,
            )
            @ utils.rotate_u(
                u=up_angles,
                device=device,
            )
        )

        east_angles, north_angles, up_angles = utils.decompose_rotations(
            initial_vector=self.initial_orientations[:, :-1],
            target_vector=self.artist_standard_orientation[:-1],
            device=device,
        )

        # Return orientation matrix multiplied by the initial orientation offset.
        return (
            orientations
            @ utils.rotate_e(
                e=east_angles,
                device=device,
            )
            @ utils.rotate_n(
                n=north_angles,
                device=device,
            )
            @ utils.rotate_u(
                u=up_angles,
                device=device,
            )
        )

    def align_surfaces_with_motor_positions(
        self,
        motor_positions: torch.Tensor,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        device: Union[torch.device, str] = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Align given surface points and surface normals according to motor positions.

        Parameters
        ----------
        incident_ray_direction : torch.Tensor
            The direction of the rays.
        surface_points : torch.Tensor
            Points on the surface of the heliostat that reflect the light.
        surface_normals : torch.Tensor
            Normals to the surface points.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The aligned surface points.
        torch.Tensor
            The aligned surface normals.
        """
        device = torch.device(device)

        orientations = self.motor_positions_to_orientations(
            motor_positions, device=device
        )
        self.orientations = orientations

        aligned_surface_points = surface_points @ orientations.transpose(1, 2)
        aligned_surface_normals = surface_normals @ orientations.transpose(1, 2)

        return aligned_surface_points, aligned_surface_normals

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
