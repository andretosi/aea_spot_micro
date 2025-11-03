from abc import ABC, abstractmethod

import numpy as np
from spotmicro.agent.command import Command

class Controller(ABC):
    def __init__(self):
        self._target_linear_velocity = np.zeros(2)
        self._target_angular_velocity = np.zeros(1)

    @abstractmethod
    def update_reference(self, *args, **kwargs) -> Command:
        """
        Set target velocities to the commanded value
        """
        pass   

    @property
    def target_linear_velocity(self) -> np.ndarray:
        """
        Get the NORMALIZED target velocity for locomotion.
        It is in the form np.ndarray([vx, vy])
        """
        return self._target_linear_velocity

    
    @property
    def target_angular_velocity(self) -> np.ndarray:
        """
        Get the NORMALIZED value for target angular velocity.
        It is just one value inside an np.ndarray representing angular velocity along the z axis
        """
        return self._target_angular_velocity
    
class RandomController(Controller):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("RandomController is not implemented yet")
    
    #TODO
    def update_reference(self, *args, **kwargs):
        pass

class MnKController(Controller):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("MnKController is not implemented yet")

    def update_reference(self, *args, **kwargs):
        pass

class joystickController(Controller):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("JoysticController is not implemented yet")
    
    def update_reference(self, *args, **kwargs):
        pass

class controllerFactory(ABC):
    """
    Class used to create different types of controllers in a standardized and configurable way.
    It is somewhat extensible (it doesn't cost much to update this) but maybe we can find a better alternative that is also automated and configurable
    """
    @staticmethod
    def createController(controller_type: str) -> Controller:
        if controller_type == "random":
            return RandomController()
        elif controller_type == "mnk":
            return MnKController()
        if controller_type == "joystick":
            return joystickController()
        else:
            raise ValueError(f"Requested controller ({controller_type}) does not exist")
        