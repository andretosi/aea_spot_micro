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
        self.shaft_id = None
        self.head_id = None
        self.cone1_id = None
        self.cone2_id = None

    def spawn(self):
        
        self.line_id = p.addUserDebugLine(
            lineFromXYZ=[0,0,0],
            lineToXYZ=[0,0,0.001],
            lineColorRGB=[1,0,0],
            lineWidth=3,
            physicsClientId=self.client_id
        )

    def update(self, position, orientation, dt:float):
        # Corpo della freccia
        if self.shaft_id is None:
            shaft_vs = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.005,
                length=0.15,
                rgbaColor=[1, 0, 0, 1],
                physicsClientId=self.client_id
            )
            self.shaft_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=shaft_vs,
                basePosition=[0.075, 0, 0.5],
                baseOrientation=p.getQuaternionFromEuler([0, np.pi/2, 0]),
                physicsClientId=self.client_id
            )

        # Anello base cono
        if self.head_id is None:
            head_vs = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.01,
                length=0.01,
                rgbaColor=[1, 0, 0, 1],
                physicsClientId=self.client_id
            )
            self.head_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=head_vs,
                basePosition=[0.15, 0, 0.5],
                baseOrientation=p.getQuaternionFromEuler([0, np.pi/2, 0]),
                physicsClientId=self.client_id
            )

        # Anello medio cono
        if self.cone1_id is None:
            cone1_vs = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.005,
                length=0.01,
                rgbaColor=[1, 0, 0, 1],
                physicsClientId=self.client_id
            )
            self.cone1_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=cone1_vs,
                basePosition=[0, 0, 0.5],
                baseOrientation=p.getQuaternionFromEuler([0, np.pi/2, 0]),
                physicsClientId=self.client_id
            )

        # Punta finale cono
        if self.cone2_id is None:
            cone2_vs = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.0025,
                length=0.005,
                rgbaColor=[1, 0, 0, 1],
                physicsClientId=self.client_id
            )
            self.cone2_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=cone2_vs,
                basePosition=[0, 0, 0.5],
                baseOrientation=p.getQuaternionFromEuler([0, np.pi/2, 0]),
                physicsClientId=self.client_id
            )

        rot_matrix = p.getMatrixFromQuaternion(orientation)
        direction = np.array([rot_matrix[0], rot_matrix[1], rot_matrix[2]])

        draw_pos = position.copy()
        draw_pos[2] = 0.5

        rot_to_x = p.getQuaternionFromEuler([0, np.pi/2, 0])
        _, shaft_orientation = p.multiplyTransforms(
            [0, 0, 0], orientation,
            [0, 0, 0], rot_to_x
        )

        # Cilindro: centro a metà lunghezza
        shaft_pos = np.array(draw_pos) - direction * 0.075
        p.resetBasePositionAndOrientation(
            self.shaft_id, shaft_pos.tolist(), shaft_orientation,
            physicsClientId=self.client_id
        )

        # Anello base: fine cilindro + metà sua lunghezza
        head_pos = np.array(draw_pos) - direction * 0.155
        p.resetBasePositionAndOrientation(
            self.head_id, head_pos.tolist(), shaft_orientation,
            physicsClientId=self.client_id
        )

        # Anello medio: fine head + metà sua lunghezza
        cone1_pos = np.array(draw_pos) - direction * 0.165
        p.resetBasePositionAndOrientation(
            self.cone1_id, cone1_pos.tolist(), shaft_orientation,
            physicsClientId=self.client_id
        )

        # Punta: fine cone1 + metà sua lunghezza
        cone2_pos = np.array(draw_pos) - direction * 0.1725
        p.resetBasePositionAndOrientation(
            self.cone2_id, cone2_pos.tolist(), shaft_orientation,
            physicsClientId=self.client_id
        )

        '''
        # Linea originale
        end = draw_pos - 0.3 * direction

        if self.line_id is not None:
            p.removeUserDebugItem(self.line_id, physicsClientId=self.client_id)

        self.line_id = p.addUserDebugLine(
            draw_pos,
            end,
            [1, 0, 0],
            3,
            lifeTime=0.3,
            physicsClientId=self.client_id
        )
        '''

    def reset(self, start_pos, start_quat):
        self.update(start_pos, start_quat, dt=0.1)