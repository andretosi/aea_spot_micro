import numpy as np
import pybullet
from dataclasses import dataclass, field
from collections import deque
from Config import Config

#questa classe contiene soltanto dei dati
#nello specifico contiene tutte informazioni relative allo stato corrente 
#dell'agent nell'ambiente. 

#@dataclass è un'etichetta che serve per rendere molto più leggibile la definizione
#degli attributi della classe

@dataclass
class AgentState:
    """
    this dataclass contains all the useful data about the current state of the agent
    """
    base_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    feet_contacts: set = field(default_factory=set)
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(0))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def roll_pitch_yaw(self):
        return pybullet.getEulerFromQuaternion(self.base_orientation)

class Joint:
    """
    All data used to define a Joint
    """
    def __init__(self, name: str, joint_id: int, joint_link_idx: int, joint_type: str, limits: tuple, config: Config):
        """
        Parameters
        ----------
        name : str
            name of the joint.
        joint_id: int
            position of the joint in the array with all the joints
        joint_link_idx: int
            internal id used by pybullet to identify the link associated with the joint
        joint_type: str
            type of the joint: shoulder, leg, foot
        limits: tuple
            (min, max) positional limits of the joint
        config: Config
            set of attributes taken from agentConfig.yaml
        """
        self.name = name
        self.leftright = name.split("_")[1]
        self.frontback = name.split("_")[0]
        self.id = joint_id
        self.link_id = joint_link_idx
        if limits[0] >= limits[1]:
            raise ValueError(f"Joint {self.name} has invalid limits: {limits}")
        self.limits = limits
        self.mid = 0.5 * (self.limits[0] + self.limits[1])
        self.type = joint_type  # shoulder, leg, foot
        self.effort = 0
        self.max_torque = config.max_torque

        # --- Type-dependent homing & gains ---
        if self.type == "shoulder":
            self.homing_position = config.left_shoulder_hp if self.leftright == "left" else config.right_shoulder_hp
            self.gain = config.shoulder_gain
            self.deadzone = config.shoulder_deadzone
            self.power = config.shoulder_power

        elif self.type == "leg":
            self.homing_position = config.front_legs_hp if self.frontback == "front" else config.rear_legs_hp
            self.gain = config.leg_gain
            self.deadzone = config.leg_deadzone
            self.power = config.leg_power

        elif self.type == "foot":
            self.homing_position = config.front_feet_hp if self.frontback == "front" else config.rear_feet_hp
            self.gain = config.foot_gain
            self.deadzone = config.foot_deadzone
            self.power = config.foot_power
    # the neural network outputs a NORALIZED vector with the action that the robot should perform.
    # This function converts the vector into a joint position, used by pybullet to move the robot.
    def from_position_to_action(self, pos: float) -> float:
        high, low = self.limits
        return (2*pos - high - low) / (high - low)

    def from_action_to_position(self, action: float) -> float:
        """Map action ∈ [-1,1] → joint position."""
        a = float(np.clip(action, -1.0, 1.0))
        low, high = self.limits
        norm_hp = self.from_position_to_action(self.homing_position)

        lin_map = lambda x, xa, ya, xb, yb: yb + (yb - ya) / (xb - xa) * (x - xb)

        if abs(a - norm_hp) < self.deadzone:
            return self.homing_position
        elif (a - norm_hp) < 0:
            return lin_map(a, -1, low, norm_hp-self.deadzone, self.homing_position)
        else:
            return lin_map(a, 1, high, norm_hp+self.deadzone, self.homing_position)


class Agent:
    """
    This class represents the Robot in the simulation. The data is taken from the pybullet simulation.
    
    Attributes
    ------------
    - _config : Config 
        contains all the data written in agentConfig.yaml
    - _state : AgentState 
        contains all the useful data about the current state of the Agent
    - _robot_id : int
        integer used by PyBullet to identify the URDF loaded in the simulation. It will be
        used to refer to the SpotMicro entity during the simulation
    - _action : npArray
        it's a vector the size of action_space_size
    - _motor_joints : tuple(Joint, ...)
        list of all the joints that can revolute of the robot.    
        It's the same size of the action vector
    - _joint_history: queue [tuple(joint_positions : npArray, joint_velocities : npArray), ...]
        Queue that holds history of joint_positions and joint_velocities.
    
    Methods
    ----------
    - reset(spawn_height: float):
        Reset agent state and simulation. Body position, orientation, 
        and joint position are set to homing in pybullet simulation.

    - apply_action(action: np.ndarray):
        This method takes as input a NORMALIZED action, maps it to joint positions and applies it
        to the joints through pyBullet. All the data about the Agent is then updated, based on the 
        new state in witch it ended up.
    
    - sync_state():
        This method updates AgentState and the joint_history. 
        This must be called after pybullet.stepSimulation() in order to update values of the Agent class.

    - _get_feet_contacts():
        This method saves which feet are touching the ground. This info is part of the state vector
    
    - _update_state():
        Query pybullet and update AgentState.
    
    - _update_joint_history(): 
        Enqueues the current joint velocities and joint positions.

    Notes
    --------
    The methods are meant to be called in this order:
    apply_action() -> pybullet.stepSimulation() -> sync_state()


    """
    def __init__(self, physics_client, config: Config, action_space_size: int, spawn_height: float):
        self._config = config
        self.physics_client = physics_client
        self._action_space_size = action_space_size

        # --- State ---
        self._state = AgentState(
            base_position=np.array([0.0, 0.0, spawn_height]),
            base_orientation=pybullet.getQuaternionFromEuler([0, self._config.homing_pitch, np.pi]),
        )
        self._action = np.zeros(self._action_space_size, dtype=np.float32)
        self._previous_action = np.zeros(self._action_space_size, dtype=np.float32)
        self._joint_history = deque(maxlen=self.config.joint_history_maxlen) # It will hold tuples with np.ndarray of joint_positions and joint_velocities

        # --- Load URDF ---
        self._robot_id = pybullet.loadURDF(
            "spotmicroai.urdf",
            basePosition=self._state.base_position,
            baseOrientation=self._state.base_orientation,
            physicsClientId=self.physics_client,
        )

        # --- Joints ---
        motor_joints = []
        homing_positions = []

        # this loop fills two arrays with this data:
        # motor_joints = [j1 : Joint,j2 : Joint, ...] is filled with objects of the Joint class (defined above), 
        # the only ones that can REVOLUTE.
        
        # homing_positions = [x1 : int, ....] is filled with the homing positions of each Joint. These
        # are the positions used to reset the position of each joint.

        for i in range(pybullet.getNumJoints(self._robot_id)):
            joint_info = pybullet.getJointInfo(self._robot_id, i)
            joint_link_id = joint_info[0]
            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_info[2]
            joint_limits = (joint_info[8], joint_info[9])

            if joint_type == pybullet.JOINT_REVOLUTE:
                joint_category = joint_name.split("_")[-1]
                joint = Joint(joint_name, i, joint_link_id, joint_category, joint_limits, self._config)
                motor_joints.append(joint)
                homing_positions.append(joint.homing_position)

        self._motor_joints = tuple(motor_joints)
        self._homing_positions = np.array(homing_positions)

        for idx, joint in enumerate(self._motor_joints):
            assert joint.id == pybullet.getJointInfo(self._robot_id, joint.id)[0], \
                f"Joint index mismatch at position {idx}"
        
        self.default_actions = np.array([j.from_position_to_action(j.homing_position) for j in self.motor_joints])


    def reset(self, spawn_heigt: float):
        """
        Reset agent state
        """

        self._state = AgentState(
            base_position=np.array([0.0, 0.0, spawn_heigt]),
            base_orientation=pybullet.getQuaternionFromEuler([0, self._config.homing_pitch, np.pi]),
            #linear_velocity=np.array([0.0, 0.0, 0.0])
            #angular_velocity=np.array([0.0, 0.0, 0.0])
        )#
        self._joint_history.clear()
        dummy_joint_state = (np.copy(self._homing_positions), np.zeros(len(self._motor_joints)))
        for _ in range(5):
            self._joint_history.append(dummy_joint_state)

        pybullet.resetBasePositionAndOrientation(
            self._robot_id,
            self._state.base_position,
            self._state.base_orientation,
            physicsClientId=self.physics_client,
        )

        pybullet.resetBaseVelocity(
            self._robot_id,
            linearVelocity=[0,0,0],
            angularVelocity=[0,0,0],
            physicsClientId=self.physics_client,
        )

        # Reset all motor joints to homing
        for i, joint in enumerate(self._motor_joints):
            pybullet.resetJointState(
                self._robot_id,
                joint.id,
                targetValue=self.homing_positions[i],
                targetVelocity=0.0,
                physicsClientId=self.physics_client,
            )

        # Reset actions to "homing" which is 0
        self._action = np.zeros(len(self._motor_joints), dtype=np.float32)
        self._previous_action = np.zeros(len(self._motor_joints), dtype=np.float32)
    
    def apply_action(self, action: np.ndarray):
        """
        This method takes as input a NORMALIZED action, maps it to joint positions and sets 
        the commands of the joints in the pybullet simulation. 
        The result of the action will be applied once stepSimulation() will be called.
        (This method takes a NORMALIZED action, updates the agent, maps
        it to joint positions and applies it to the joints through pybullet)
        """
        self._previous_action = self._action.copy()
        self._action = action
        
        # this loop updates the position of all the movable joints of the robot based
        # on the target positions computed by the from_action_to_position method.
        # The max_torque of the joints is defined in the agentConfig file.

        # setJointMotorControl2 doesn't actually "move" the robot in the simulation,
        # but tells the motor what to do once stepSimulation() will be called.
        for i, joint in enumerate(self._motor_joints):
            pybullet.setJointMotorControl2(
                bodyUniqueId = self._robot_id,
                jointIndex = joint.id,
                controlMode = pybullet.POSITION_CONTROL,
                targetPosition = joint.from_action_to_position(action[i]),
                force = joint.max_torque
            )

        return

    def sync_state(self):
        self._update_state()
        self._update_joint_history()

    def _get_feet_contacts(self) -> set:
        """
        This method saves which feet are touching the ground (part of the state vector)
        returns a set of link indices of the feet in contact with the ground
        """
        contact_points = pybullet.getContactPoints(
            bodyA=self._robot_id,
            bodyB=self._plane_id,
            physicsClientId=self.physics_client
        )

        feet_in_contact = set()

        for contact in contact_points:
            link_idx = contact[3]  # linkIndexA from your robot
            for joint in self._motor_joints:
                if link_idx - 1 == joint.link_id and joint.type == "foot": # linkd indices in contacts are shifted by 1 compared to the ones stored in the joint objects (it's conventional). We apply the -1 shift to address the joint with their saved link_id
                    feet_in_contact.add(link_idx - 1)
        
        return feet_in_contact

    def _update_state(self):
        """Query pybullet and update AgentState."""

        #gets from pybullet data about position, orientation, velocity
        pos, ori = pybullet.getBasePositionAndOrientation(self._robot_id)
        lin_vel, ang_vel = pybullet.getBaseVelocity(self._robot_id)

        #gets from pybullet position, velocity and effort of each joint
        joint_positions = []
        joint_velocities = []
        for joint in self._motor_joints:
            state = pybullet.getJointState(self._robot_id, joint.id)
            joint_positions.append(state[0])
            joint_velocities.append(state[1])
            joint.effort = state[3]

        #updates the agent state with all the new data
        self._state.base_position = np.array(pos)
        self._state.base_orientation = np.array(ori)
        self._state.linear_velocity = np.array(lin_vel)
        self._state.angular_velocity = np.array(ang_vel)
        self._state.joint_positions = np.array(joint_positions)
        self._state.joint_velocities = np.array(joint_velocities)
        self._state.feet_contacts = self._get_feet_contacts

    def _update_joint_history(self):
        self._joint_history.append((self._state.joint_positions, self._state.joint_velocities))

    #this methods return the values of some of the class attributes
    # --- Accessors ---
    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def config(self) -> Config:
        return self._config

    @property
    def agent_id(self):
        return self._robot_id

    @property
    def previous_action(self) -> np.ndarray:
        return self._previous_action

    @previous_action.setter
    def previous_action(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray) and len(value) == self._action_space_size:
            self._previous_action = value
        else:
            raise ValueError(f"previous_action must be a numpy array of length {self._action_space_size}")

    @property
    def action(self) -> np.ndarray:
        return self._action
    
    @property
    def joint_history(self) -> deque:
        return self._joint_history
    
    @joint_history.setter
    def joint_history(self, history: deque):
        if isinstance(history, deque) and len(history) <= self._config.joint_history_maxlen:
            self._joint_history = history
    
    @property
    def homing_positions(self) -> np.ndarray:
        return self._homing_positions
    
    @property
    def motor_joints(self) -> tuple:
        return self._motor_joints
