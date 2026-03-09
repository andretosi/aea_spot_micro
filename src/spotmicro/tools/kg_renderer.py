from abc import ABC, abstractmethod
import pybullet as p
import numpy as np

class KG_Renderer(ABC):

    @abstractmethod
    def __init__(self, client_id=None):
        pass

    @abstractmethod
    def spawn(self):
        pass

    @abstractmethod
    def update(self, position, orientation, dt:float):
        pass

    @abstractmethod
    def reset(self, start_pos=[0,0,0], start_quat=[0,0,0,1]):
        pass



class PyBulletRenderer(KG_Renderer):

    def __init__(self, client_id):

        self.client_id = client_id
        self.line_id = None

    def spawn(self):
        self.line_id = p.addUserDebugLine(
            lineFromXYZ=[0,0,0],
            lineToXYZ=[0,0,0.001],
            lineColorRGB=[1,0,0],
            lineWidth=3,
            physicsClientId=self.client_id
        )
    
    def update(self, position, orientation, dt:float):
        """
        Aggiorna posizione e orientamento del ghost
        """
        # calcola direzione della freccia dal quaternion
        rot_matrix = p.getMatrixFromQuaternion(orientation)
        direction = np.array([rot_matrix[0], rot_matrix[1], rot_matrix[2]])  # asse x

        draw_pos = position.copy()
        draw_pos[2] = 0.5  # altezza fissa sopra il terreno

        end = draw_pos - 0.3 * direction

        self.line_id = p.addUserDebugLine(
            draw_pos,
            end,
            [1, 0, 0],
            3,
            replaceItemUniqueId=self.line_id,
            lifeTime=1,
            physicsClientId=self.client_id
        ) 

    def reset(self, start_pos, start_quat):
        """
        Reset del ghost
        """
        self.update(start_pos, start_quat, dt=0.1)