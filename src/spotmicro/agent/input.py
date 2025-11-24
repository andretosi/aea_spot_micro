import numpy as np
from dataclasses import dataclass

@dataclass
class Input:
    """
    This utility class represents a command issued to the controller. We could have used an array, but this is much clearer and also defines a stable interface
    """
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y , self.w])