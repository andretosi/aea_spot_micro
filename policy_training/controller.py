import numpy as np
from command import Command

class Controller:
    def __init__(self, max_linear_velocity: float, max_angular_velocity: float):
        self._TARGET_LINEAR_VELOCITY = np.zeros(3)
        self._TARGET_ANGULAR_VELOCITY = np.zeros(3)

        assert max_linear_velocity > 0, "Invalid (negative) max linear velocity error"
        self._max_lin_vel = max_angular_velocity
        assert max_angular_velocity > 0, "Invalid (negative) max angular velocity"
        self._max_ang_vel = max_angular_velocity


    def update_reference(self, cmd: Command):
        cmd_arr = cmd.as_array
        assert np.any((cmd_arr < -1.0) | (cmd_arr > 1.0)), f"Command ({cmd_arr}) out of range"

        #How to translate to reference values?
        #The controller needs to now max lin and ang velocity
        #Since they are defined in a config, and we do not expect to change them at runtime (right?) we can directly build it with that
        
        

    @property
    def target_lin_velocity(self) -> np.ndarray:
        """
        Get the current target direction for locomotion (unit vector).
        """
        return self._TARGET_LINEAR_VELOCITY

    
    @property
    def target_ang_velocity(self) -> np.ndarray:
        """
        Get the current target direction for locomotion (unit vector).
        """
        return self._TARGET_ANGULAR_VELOCITY