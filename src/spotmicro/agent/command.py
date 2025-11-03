import numpy as np
from dataclasses import dataclass

@dataclass
class Command:
    """
    This utility class represents a command issued to the controller. We could have used an array, but this is much clearer and also defines a stable interface
    """
    x: float
    y: float
    theta: float