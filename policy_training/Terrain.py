import numpy as np
import pybullet
from scipy.ndimage import gaussian_filter

class Terrain:
    """
    MAIN IDEA: structure this as to take input for curriculum and params from a config, so that every config file (terrain, robot, function) can be "immutably" packeted and reused for the sake of documentation
    """
    def __init__(self, physics_client, config: dict):
        self.physics_client = physics_client
        self.env_config = config
        self.terrain_id = None
        self.difficulty = 0.0
        self.flat = self.env_config.use_flat