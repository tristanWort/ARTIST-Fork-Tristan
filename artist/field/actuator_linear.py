from typing import Union, Tuple

import torch

from artist.field.actuator import Actuators


class LinearActuators(Actuators):
    """
    Implements the behavior of a linear actuators.

    Attributes
    ----------
    clockwise_axis_movements : bool
        Turning directions of the joints.
    increments : torch.Tensor
        The stroke length changes per motor step.
    initial_stroke_lengths : torch.Tensor
        The stroke lengths for a motor step of 0.
    offsets : torch.Tensor
        The offsets between the linear actuators' pivoting points and the points
        around which the actuators are allowed to pivot.
    pivot_radii : torch.Tensor
        The actuators' pivoting radii.
    initial_angle : torch.Tensor
        The angles that the actuators introduce to the manipulated coordinate systems
        at the initial stroke lengths.

    Methods
    -------
    motor_positions_to_angles()
        Calculate the joint angles for given motor positions.
    angles_to_motor_positions()
        Calculate the motor positions for given joint angles.
    forward()
        Specify the forward pass.

    See Also
    --------
    :class:`Actuator` : Reference to the parent class.
    """

    def __init__(
        self,
        clockwise_axis_movements: torch.nn.ParameterList,
        increments: torch.nn.ParameterList,
        initial_stroke_lengths: torch.nn.ParameterList,
        offsets: torch.nn.ParameterList,
        pivot_radii: torch.nn.ParameterList,
        initial_angles: torch.nn.ParameterList,
    ) -> None:
        """
        Initialize linear actuators.

        A linear actuator describes movement within a 2D plane. The clockwise axis movement attribute describes 
        the turning direction of the actuator. The linear actuator is further parametrized by five parameters.
        These are the increment, which stores the information about the stroke length change per motor step, 
        the initial stroke length, and an offset that describes the difference between the linear actuator's 
        pivoting point and the point around which the actuator is allowed to pivot. Next, the actuator's pivoting 
        radius is described by the pivot radius and lastly, the initial angle indicates the angle that the 
        actuator introduces to the manipulated coordinate system at the initial stroke length.

        Parameters
        ----------
        clockwise_axis_movements : torch.Tensor
            Turning directions of the joints.
        increments : torch.nn.ParameterList
            The stroke length changes per motor step.
        initial_stroke_lengths : torch.nn.ParameterList
            The stroke lengths for a motor step of 0.
        offsets : torch.nn.ParameterList
            The offsets between the linear actuators' pivoting points and the points
            around which the actuators are allowed to pivot.
        pivot_radii : torch.nn.ParameterList
            The actuators' pivoting radii.
        initial_angle : torch.nn.ParameterList
            The angles that the actuators introduce to the manipulated coordinate systems
            at the initial stroke lengths.
        """
        super().__init__()
        # Todo clock
        self.clockwise_axis_movements = clockwise_axis_movements
        self.increments = increments
        self.initial_stroke_lengths = initial_stroke_lengths
        self.offsets = offsets
        self.pivot_radii = pivot_radii
        self.initial_angles = initial_angles

    def _get_stacked_parameters(
        self, 
        device: Union[torch.device, str] = "cuda"
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                   torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all actuator parameters as stacked tensors.
        The stacked tensors maintain references to the original parameters.
        
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
              torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of stacked parameters.
        """
        # Stack parameters while maintaining the references
        clockwise_axis_movements = torch.stack(
            [torch.stack([param for param in hel]) for hel in self.clockwise_axis_movements]
            ).to(device)
        
        increments = torch.stack(
            [torch.stack([param for param in hel]) for hel in self.increments]
            ).to(device)
        
        initial_stroke_lengths = torch.stack(
            [torch.stack([param for param in hel]) for hel in self.initial_stroke_lengths]
            ).to(device)
        
        offsets = torch.stack(
            [torch.stack([param for param in hel]) for hel in self.offsets]
            ).to(device)
        
        pivot_radii = torch.stack(
            [torch.stack([param for param in hel]) for hel in self.pivot_radii]
            ).to(device)
        
        initial_angles = torch.stack(
            [torch.stack([param for param in hel]) for hel in self.initial_angles]
            ).to(device)
        
        return clockwise_axis_movements, increments, initial_stroke_lengths, offsets, pivot_radii, initial_angles
    
    def _motor_positions_to_absolute_angles(
        self, motor_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert motor steps into angles using actuator geometries.

        Calculate absolute angles based solely on the motors' current positions and the geometries
        of the actuators. This gives the angles of the actuators in a global sense. It does not
        consider the starting positions of the motors.

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.

        Returns
        -------
        torch.Tensor
            The calculated absolute angles.
        """
        device = motor_positions.device
        _, increments, initial_stroke_lengths, offsets, pivot_radii, _ = self._get_stacked_parameters(device=device)
        
        # see page 17, MA Leibauer
        stroke_lengths = motor_positions / increments + initial_stroke_lengths
        calc_step_1 = offsets**2 + pivot_radii**2 - stroke_lengths**2
        calc_step_2 = 2.0 * offsets * pivot_radii
        calc_step_3 = calc_step_1 / calc_step_2
        absolute_angles = torch.arccos(calc_step_3)
        return absolute_angles

    def motor_positions_to_angles(
        self, motor_positions: torch.Tensor, device: Union[torch.device, str] = "cuda"
    ) -> torch.Tensor:
        """
        Calculate the joint angles for given motor positions.

        The absolute angles are adjusted to be relative to the initial angles. This accounts for
        the initial angles and the motors' directions (clockwise or counterclockwise).

        Parameters
        ----------
        motor_positions : torch.Tensor
            The motor positions.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The joint angles corresponding to the motor positions.
        """
        device = torch.device(device)
        absolute_angles = self._motor_positions_to_absolute_angles(
            motor_positions=motor_positions
        )
        absolute_initial_angles = self._motor_positions_to_absolute_angles(
            motor_positions=torch.zeros_like(motor_positions, device=device)
        )
        delta_angles = absolute_initial_angles - absolute_angles

        clockwise_axis_movements, _, _, _, _, initial_angles = self._get_stacked_parameters(device=device)
        relative_angles = (
            initial_angles + delta_angles * (clockwise_axis_movements == 1) - delta_angles * (clockwise_axis_movements == 0)
        )
        return relative_angles

    def angles_to_motor_positions(
        self, angles: torch.Tensor, device: Union[torch.device, str] = "cuda"
    ) -> torch.Tensor:
        """
        Calculate the motor positions for given joint angles.

        First the relative angular changes are calculated based on the given angles.
        Then the corresponding stroke lengths are determined using trigonometric
        relationships. These stroke lengths are converted into motor steps.

        Parameters
        ----------
        angles : torch.Tensor
            The joint angles.
        device : Union[torch.device, str]
            The device on which to initialize tensors (default is cuda).

        Returns
        -------
        torch.Tensor
            The motor steps.
        """
        device = torch.device(device)
        
        clockwise_axis_movements, increments, initial_stroke_lengths, offsets, pivot_radii, initial_angles = self._get_stacked_parameters(device=device)
        
        delta_angles = torch.where(
            clockwise_axis_movements == 1, 
            angles - initial_angles,
            initial_angles - angles
        )

        absolute_initial_angles = self._motor_positions_to_absolute_angles(
            motor_positions=torch.zeros_like(angles, device=device)
        )
        initial_angles = absolute_initial_angles - delta_angles

        calc_step_3 = torch.cos(initial_angles)
        calc_step_2 = 2.0 * offsets * pivot_radii
        calc_step_1 = calc_step_3 * calc_step_2
        stroke_lengths = torch.sqrt(offsets**2 + pivot_radii**2 - calc_step_1)
        motor_positions = (stroke_lengths - initial_stroke_lengths) * increments
        return motor_positions

    def forward(self) -> None:
        """
        Specify the forward pass.

        Raises
        ------
        NotImplementedError
            Whenever called.
        """
        raise NotImplementedError("Not Implemented!")
