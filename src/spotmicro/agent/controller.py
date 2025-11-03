import numpy as np
from spotmicro.agent.command import Command
from spotmicro.agent.agent import Agent

class Controller:
    def __init__(
            self, 
            max_forward_linear_velocity: float, 
            max_lateral_linear_velocity: float, 
            max_angular_velocity: float
            ):
        self._target_linear_velocity = np.zeros(3)
        self._target_angular_velocity = np.zeros(3)

        assert max_forward_linear_velocity > 0, "Invalid (negative) max forward linear velocity error"
        self._max_forward_lin_vel = max_forward_linear_velocity
        assert max_lateral_linear_velocity > 0, "Invalid (negative) max lateral linear velocity error"
        self._max_lateral_lin_vel = max_lateral_linear_velocity
        assert max_angular_velocity > 0, "Invalid (negative) max angular velocity"
        self._max_ang_vel = max_angular_velocity


    def update_reference(self, cmd: Command):
        """
        Given a command (of type Command) referred to the robot frame (its perspective) and update reference velocities IN ROBOT FRAME accordingly
        """
        # Safety checks
        assert -1.0 <= cmd.x <= 1.0, f"command.x out of range (got {cmd.x})"
        assert -1.0 <= cmd.y <= 1.0, f"command.y out of range (got {cmd.y})"
        assert -1.0 <= cmd.theta <= 1.0, f"command.rot out of range (got {cmd.theta})"

        #How to translate to reference values?
        #The following code only makes sense if references are giiven in the robot's frame (so, from it's own perspective)
        
        #TODO: maybe reduce dimensionality and don't use useless directions?
        self._target_linear_velocity = np.array(
            [
                cmd.x * self._max_forward_lin_vel,
                cmd.y * self._max_lateral_lin_vel,
                0.0
            ]
        )
        self._target_angular_velocity = np.array(
            [
                0.0,
                0.0,
                cmd.theta * self._max_ang_vel
            ]
        )
        
        

    @property
    def target_linear_velocity(self) -> np.ndarray:
        """
        Get the current target direction for locomotion (unit vector).
        """
        return self._TARGET_LINEAR_VELOCITY

    
    @property
    def target_angular_velocity(self) -> np.ndarray:
        """
        Get the current target direction for locomotion (unit vector).
        """
        return self._TARGET_ANGULAR_VELOCITY