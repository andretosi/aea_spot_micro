import numpy as np
import pybullet
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
from Config import Config

class Terrain:
    def __init__(self, physics_client, config: Config):
        self.physics_client = physics_client
        self._config = config
        self._terrain_id = None
        self._tilt_step = 0

    def generate(self):
        if self._config.mode == "flat":
            self._terrain_id = pybullet.loadURDF(
                "plane.urdf",
                physicsClientId=self.physics_client
            )

        elif self._config.mode == "tilting":
            self._terrain_id = pybullet.loadURDF(
                "plane.urdf",
                physicsClientId=self.physics_client
            )
            self._tilt_step = 0

        elif self._config.mode == "irregular":
            self._terrain_id = self._create_terrain()

        else:
            raise ValueError(f"Unknown terrain mode: {self._config.mode}")

        return self._terrain_id

    def reset(self):
        self._tilt_step = 0
        self._tilt_phase = np.random.uniform(0, 2 * np.pi)
        if self._config.evolving:
            pass #@TODO
    
    def tilt_plane(self):
        
        self._tilt_step += 1
        freq = self._config.rotation_frequency
        max_angle = self._config.max_tilt_angle

        tilt_x = max_angle * np.sin(freq * self._tilt_step + self._tilt_phase)
        tilt_y = max_angle * np.cos(freq * self._tilt_step + self._tilt_phase)

        quat = pybullet.getQuaternionFromEuler([tilt_x, tilt_y, 0])

        pybullet.resetBasePositionAndOrientation(
            self._terrain_id,
            posObj=[0, 0, 0],
            ornObj=quat,
            physicsClientId=self.physics_client
        )

    
    def _gen_pothole_heightfield(self):
        rows = cols = self._config.grid_size
        cell = self._config.cell_size
        H = np.zeros((rows, cols), dtype=np.float32)

        xs = np.arange(cols) * cell
        ys = np.arange(rows) * cell
        X, Y = np.meshgrid(xs, ys)

        rng = np.random.default_rng(42)
        cx = rng.uniform(xs.min()+self._config.pit_radius, xs.max()-self._config.pit_radius, size=self._config.n_pits)
        cy = rng.uniform(ys.min()+self._config.pit_radius, ys.max()-self._config.pit_radius, size=self._config.n_pits)

        r2 = self._config.pit_radius**2
        for x0, y0 in zip(cx, cy):
            d2 = (X - x0)**2 + (Y - y0)**2
            mask = d2 < r2
            w = 1.0 - np.sqrt(d2[mask]) / self._config.pit_radius
            H[mask] -= self._config.pit_depth * (w**2)

        if self._config.smooth_sigma_cells and self._config.smooth_sigma_cells > 0:
            H = gaussian_filter(H, sigma=self._config.smooth_sigma_cells)

        return H

    def _create_terrain(self):
        self._config = self._config
        rows = cols = self._config.grid_size
        cell = self._config.cell_size

        y = np.linspace(0, cols*cell, cols)
        Y = np.tile(y, (rows, 1))
        ridges = self._config.ridges_height * np.sin(2*np.pi * Y / self._config.ridges_distance)

        H = self._gen_pothole_heightfield()
        H += ridges

        H = np.clip(H, self._config.min_height_bumps, self._config.max_height_bumps)
        heightfield_data = H.flatten().tolist()

        shape = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_HEIGHTFIELD,
            meshScale=[cell, cell, 1.0],
            heightfieldTextureScaling=(rows - 1) / 2,
            heightfieldData=heightfield_data,
            numHeightfieldRows=rows,
            numHeightfieldColumns=cols,
            physicsClientId=self.physics_client
        )

        terrain_id = pybullet.createMultiBody(0, shape)
        return terrain_id

    @property
    def terrain_id(self):
        return self._terrain_id

    @property
    def config(self):
        return self._config