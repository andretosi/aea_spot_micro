import pybullet, pybullet_data
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
import inspect, os, pickle, warnings

class Joint:
    def __init__(self, name: str, joint_id: int, joint_link_idx: int, joint_type: str, limits: tuple):
        self.name = name
        self.leftright = name.split("_")[1]
        self.frontback = name.split("_")[0]
        self.id = joint_id
        self.link_id = joint_link_idx
        if limits[0] >= limits[1]:
            raise ValueError(f"Joint {self.name} has invalid limits: {limits}")
        self.limits = limits
        self.mid = 0.5 * (self.limits[0] + self.limits[1])
        self.type = joint_type # shoulder, leg, foot
        self.effort = 0

        if self.type == "shoulder": # Can range from -0.548 to 0.548, where negatives move outwards and positives move inwards
            self.max_torque = 5 # Max value would be 6.81 
            if self.leftright == "left":
                self.homing_position = -0.05
            else:
                self.homing_position = 0.05
            self.range = 0.4
    
        elif self.type == "leg": # Can range from -2.666 to 1.548, where -2.666 is the max extension and 1.548 is the max flexion
            self.max_torque = 5
            if self.frontback == "front":
                self.homing_position = -0.4
            else:
                self.homing_position = -0.4
            self.range = 0.9

        elif self.type == "foot": # Can range from -0.1 to 2.59, where -0.1 is the max extension and 2.59 is the max flexion
            self.max_torque = 5
            if self.frontback == "front":
                self.homing_position = 1
            else:
                self.homing_position = 0.93
            self.range = 0.6
    
    def from_action_to_position(self, action: float) -> float:
        return self.homing_position + self.range * action

class SpotmicroEnv(gym.Env):
    def __init__(self, use_gui=False, reward_fn=None, init_custom_state=None, dest_save_file=None, src_save_file=None, writer=None):
        super().__init__()

        self._OBS_SPACE_SIZE = 94
        self._ACT_SPACE_SIZE = 12
        self._MAX_EPISODE_LEN = 3000
        self._TARGET_DIRECTION = np.array([1.0, 0.0, 0.0])
        self.TARGET_HEIGHT = 0.230
        self._SURVIVAL_REWARD = 3.0
        self._SIM_FREQUENCY = 240
        self._CONTROL_FREQUENCY = 60
        self._JOINT_HISTORY_MAX_LEN = 5
        self.HOMING_PITCH = -0.07

        self._episode_step_counter = 0
        self._total_steps_counter = 0

        self._robot_id = None
        self._plane_id = None
        self._motor_joints = None
        self._joint_history = deque(maxlen=self._JOINT_HISTORY_MAX_LEN)
        self._previous_action = np.zeros(self._ACT_SPACE_SIZE, dtype=np.float32)
        self.physics_client = None
        self.use_gui = use_gui
        self.np_random = None

        self._episode_reward_info = None
        self.writer = writer

        #Declaration of observation and action spaces
        self.observation_space = gym.spaces.Box(
            low = -np.inf, 
            high = np.inf, 
            shape = (self._OBS_SPACE_SIZE,), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low = -1.0, 
            high = 1.0, 
            shape = (self._ACT_SPACE_SIZE,),
            dtype = np.float32
        )

        self._agent_state = {
            "base_position": None,
            "base_orientation": None,
            "linear_velocity": None,
            "angular_velocity": None,
            "ground_feet_contacts": None
        }

        self._custom_state = dict()

        #If the agents is in this state, we terminate the simulation. Should quantize the fact that it has fallen, maybe a threshold?
        self._target_state = {
            "min_height": 0.13, #meters?
            "max_height": 0.40,
            "max_pitchroll": np.radians(55)
        }

        if reward_fn is None:
            raise ValueError("reward_fn cannot be None. Provide a valid rewad function")
        elif not callable(reward_fn):
            raise ValueError("reward_fn must be callable (function)")

        self._reward_fn = reward_fn

        # Initialize custom function if provided
        if init_custom_state is not None:
            if not callable(init_custom_state):
                raise ValueError("init_custom_fn must be callable (function)")
            self._init_custom_state = init_custom_state
        else:
            self._init_custom_state = lambda env: None
    
        self._dest_save = dest_save_file
        if self._dest_save is not None:
            if not isinstance(self._dest_save, str):
                raise TypeError("Destination file path must be a string.")
            if os.path.exists(dest_save_file):
                warnings.warn(f"File '{self._dest_save}' already exists and will be overwritten.", UserWarning)
            if not self._dest_save.endswith(".pkl"):
                raise ValueError("Expected a .pkl file for environment state save destination")
            

        self._src_file = src_save_file
        if self._src_file is not None:
            if not isinstance(self._src_file, str):
                raise TypeError("Source file path must be a string.")
            if not os.path.exists(self._src_file):
                raise FileNotFoundError(f"No file found at {self._src_file}")
            if not src_save_file.endswith(".pkl"):
                raise ValueError("Expected a .pkl file for environment state save source")
            
            self.load_state()
            
    def save_state(self):
        state = {
            "total_steps_counter": self._total_steps_counter,
            "previous_action": self._previous_action,
            "joint_history": list(self._joint_history),
            "target_direction": self._TARGET_DIRECTION
        }

        with open(self._dest_save, "wb") as f:
            pickle.dump(state, f)

    def load_state(self):
        with open(self._src_file, 'rb') as f:
            state = pickle.load(f)
        
        self._total_steps_counter = state["total_steps_counter"]
        self._previous_action = state["previous_action"]
        self._joint_history = deque(state["joint_history"], maxlen=self._JOINT_HISTORY_MAX_LEN)
        self._TARGET_DIRECTION = state["target_direction"]
    
    def _is_state_initialized(self, key: str):
        value = self._agent_state.get(key)
        if value is None:
            raise ValueError(f"'{key}' has not been initialized (is currently None)")
        return value

    @property
    def agent_base_position(self) -> tuple[float, float, float]:
        """
        Returns the coordinates of the base of the agent in the form (x,y,z)
        """
        base_pos = self._is_state_initialized("base_position")
        return tuple(base_pos)
    
    @property
    def agent_base_orientation(self) -> tuple[float, float, float, float]:
        """
        Returns the quaternion representign the orientation of the base, in the form (x, y, z, w)
        """
        base_ori = self._is_state_initialized("base_orientation")
        return tuple(base_ori)
    
    @property
    def agent_linear_velocity(self) -> tuple[float, float, float]:
        """
        Returns the vector representign the linear velocity of the agent, in the form (vx, vy, vz)
        """
        lin_vel = self._is_state_initialized("linear_velocity")
        return tuple(lin_vel)
    
    @property
    def agent_angular_velocity(self) -> tuple[float, float, float]:
        """
        Returns the vector representing the angular velocity of the agent, in the form (wx, wy, wz)
        """
        ang_vel = self._is_state_initialized("angular_velocity")
        return tuple(ang_vel)
    
    @property
    def agent_ground_feet_contacts(self) -> set:
        """
        Returns a set of ids of the feet of the agent making contact with the ground
        """
        gf_contacts = self._is_state_initialized("ground_feet_contacts")
        return gf_contacts

    @property
    def agent_previous_action(self) -> np.ndarray:
        """
        Returns the action taken by the agent in the previous step
        """
        return self._previous_action
    
    @property
    def agent_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple of two arrays: the first contains the positions of each joint, the second their velocities
        """
        pos, vel = self._get_joint_states()
        return (pos, vel)
    
    @property
    def target_direction(self) -> np.ndarray:
        """
        Get the current target direction for locomotion (unit vector).
        """
        return self._TARGET_DIRECTION
    
    @target_direction.setter
    def target_direction(self, direction: tuple[float, float, float]) -> None:
        """
        Set a new target direction for locomotion. Should be a normalized 3D vector
        """
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Target direction cannot be a zero vector")
        self._TARGET_DIRECTION = np.array(np.array(direction) / norm)
    
    @property
    def num_steps(self) -> int:
        """
        Return the current number of steps
        """
        return self._total_steps_counter

    @property
    def motor_joints(self) -> tuple:
        """
        Return the list of movable joints (a list of Joint objects)
        """
        if self._motor_joints is None:
            raise ValueError("List of movable joint is not initialized yet. Yous hould call .reset() at least once before trying to access this attribute")
        return self._motor_joints

    def get_custom_state(self, key: str):
        """
        Returns the value corresponding to the given key inside a dictionary of custom variables.
        """
        if key not in self._custom_state.keys():
            raise ValueError(f"No value was initialized for the given key ({key})")
        return self._custom_state[key]
    
    
    def set_custom_state(self, key: str, value):
        """
        If given a value, it instead changes the value of the given key to the provided value
        """
        self._custom_state[key] = value

    def close(self):
        """
        Method exposed and used by SB3.
        Cleans up the simulation, saves the state if a destination path is provided
        """
        if self.physics_client is not None:
            pybullet.disconnect(self.physics_client)
            self.physics_client = None

        if self._dest_save is not None:
            self.save_state()

    def reset(self, seed: int |None = None, options: dict | None = None) -> tuple[gym.spaces.Box, dict]:
        """
        Method exposed and called by SB3 before starting each episode.
        Sets the all parameters and puts the agent in place
        """
        
        super().reset(seed=seed)
        self._episode_step_counter = 0
        self._action_counter = 0
        self._agent_state["base_position"] = (0.0 , 0.0, 0.235) #Height set specifically through trial and error
        self._agent_state["base_orientation"] = pybullet.getQuaternionFromEuler([0, self.HOMING_PITCH, np.pi])
        self._agent_state["linear_velocity"] = np.zeros(3)
        self._agent_state["angular_velocity"] = np.zeros(3)
        self._agent_state["ground_feet_contacts"] = set()
        self._previous_action = np.zeros(self._ACT_SPACE_SIZE, dtype=np.float32)

        self._episode_reward_info = []

        self._tilt_step = 0
        self._tilt_phase = np.random.uniform(0, 2 * np.pi)

        self._joint_history.clear()
        dummy_joint_state = [0.0] * 24
        for _ in range(5):
            self._joint_history.append(dummy_joint_state)

        #Initialize pybullet
        if self.physics_client is None:
            self.physics_client = pybullet.connect(pybullet.GUI if self.use_gui else pybullet.DIRECT)

        pybullet.resetSimulation(physicsClientId=self.physics_client)
        pybullet.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        pybullet.setTimeStep(1/self._SIM_FREQUENCY, physicsClientId=self.physics_client)

        #load robot URDF here
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = pybullet.loadURDF(
            "plane.urdf", 
            basePosition = [0,0,0],
            baseOrientation = pybullet.getQuaternionFromEuler([0,0,0]),
            physicsClientId=self.physics_client
        )

        pybullet.changeDynamics(
            bodyUniqueId=self._plane_id,
            linkIndex=-1,
            lateralFriction=1.0,           # <- good default
            spinningFriction=0.0,
            rollingFriction=0.0,
            restitution=0.0,
            physicsClientId=self.physics_client
        )
        
        self._robot_id = pybullet.loadURDF(
            "spotmicroai.urdf",
            basePosition = self._agent_state["base_position"],
            baseOrientation = self._agent_state["base_orientation"],
            physicsClientId = self.physics_client
        )

        # Builld the list of movable joints and assign all attributes
        if self._motor_joints is None:
            motor_joints = []
            for i in range(pybullet.getNumJoints(self._robot_id)):
                joint_info = pybullet.getJointInfo(self._robot_id, i)
                joint_link_id = joint_info[0]
                joint_name = joint_info[1].decode("utf-8")
                joint_type = joint_info[2]
                joint_limits = (joint_info[8], joint_info[9])

                if joint_type == pybullet.JOINT_REVOLUTE:
                    joint_category = joint_name.split("_")[-1]
                    motor_joints.append(Joint(joint_name, i, joint_link_id, joint_category, joint_limits))

            self._motor_joints = tuple(motor_joints) # Made immutable to avoid problems

        # Setting homing position and friction
        for joint in self._motor_joints:
            pybullet.resetJointState(self._robot_id, joint.id, joint.homing_position)
            if joint.type == "foot":
                pybullet.changeDynamics(
                    self._robot_id,
                    linkIndex=joint.id,
                    lateralFriction=1.25,
                    physicsClientId=self.physics_client
                )

        # Initialize custom state
        self._init_custom_state(self)

        #this is just to let the physics stabilize? -> might need to remove this loop
        for _ in range(10):
            pybullet.stepSimulation(physicsClientId=self.physics_client)
        self._update_agent_state()

        sig = inspect.signature(self._reward_fn)
        if len(sig.parameters) != 2:
            raise ValueError("reward_fn must accept exactly 2 parameters (env, action)")
            
        # Test reward function return type
        try:
            dummy_action = np.zeros(self._ACT_SPACE_SIZE)
            reward, info = self._reward_fn(self, dummy_action)
            if not isinstance(reward, (int, float)):
                raise ValueError("reward_fn must return a number as first return value")
            if not isinstance(info, dict):
                raise ValueError("reward_fn must return a dict as second return value")
        except Exception as e:
            raise ValueError(f"Error testing reward_fn: {str(e)}")
        

        observation = self._get_observation()
        return (observation, self._get_info())
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> tuple[gym.spaces.Box, float, bool, bool, dict]:
        """
        Method exposed and used bby SB3 to execute one time step within the environment.

        Args:
            action (gym.spaces.Box): The action to take in the environment.

        Returns:
            tuple containing
                - observation (np.ndarray): Agent's observation of the environment.
                - reward (float): Amount of reward returned after previous action.
                - terminated (bool): Whether the episode naturally ended.
                - truncated (bool): Whether the episode was artificially terminated.
                - info (dict): Contains auxiliary diagnostic information.
        """
        #Slow down the control loop
        if self._action_counter == int(self._SIM_FREQUENCY / self._CONTROL_FREQUENCY): # apply new action
            observation = self._step_simulation(action)
            self._update_history()
            self._action_counter = 0
            reward, reward_info = self._calculate_reward(action)
            self._previous_action = action.copy()
        else:                                                                         # reuse last action
            self._action_counter += 1
            observation = self._step_simulation(self._previous_action)
            reward, reward_info = self._calculate_reward(self._previous_action)

        terminated, term_penalty = self._is_target_state(self._agent_state) # checks wether the agent has fallen or not
        truncated = self._is_terminated()
        info = self._get_info()

        self._episode_reward_info.append(reward_info)
        if truncated:
            reward += self._SURVIVAL_REWARD
        if terminated:
            reward += term_penalty

        self._total_steps_counter += 1

        return observation, reward, terminated, truncated, info
    
    def plot_reward_components(self):
        keys = self._episode_reward_info[0].keys()
        values = {k: [] for k in keys}

        for step_info in self._episode_reward_info:
            for k in keys:
                values[k].append(step_info[k])

        plt.figure(figsize=(12, 6))
        for k in keys:
            plt.plot(values[k], label=k)

        plt.title("Reward Components Over Episode")
        plt.xlabel("Timestep")
        plt.ylabel("Reward Contribution")
        plt.legend()
        plt.grid(True)
        plt.savefig("plot.png")
        plt.close()
    
    def log_rewards(self, reward_dict: dict):
        if self.writer is None:
            return
        
        for key, value in reward_dict.items():
            try:
                self.writer.add_scalar(f"reward_components/{key}", value, self.num_steps)
            except Exception as e:
                print(f"[Logging Error] Could not log {key}: {e}")
    
    #@TODO: should also tilt the plane, look up the code fromm the notebook
    def _step_simulation(self, action: np.ndarray) -> np.ndarray:
        """
        Private method that calls the API to execute the given action in PyBullet.
        It should sinchronize the state of the agent in the simulation with the state recorded here!
        Accepts an action and returns an observation
        """
        #@TODO: add max torque
        # Execute the action in pybullet
        for i, joint in enumerate(self._motor_joints):
            pybullet.setJointMotorControl2(
                bodyUniqueId = self._robot_id,
                jointIndex = joint.id,
                controlMode = pybullet.POSITION_CONTROL,
                targetPosition = joint.from_action_to_position(action[i]),
                force = joint.max_torque
            )
        
        self._episode_step_counter += 1 #updates the step counter (used to check against timeouts)
        pybullet.stepSimulation()

        self._update_agent_state()

        return self._get_observation()
    
    def _get_joint_states(self) -> tuple[list[float], list[float]]:
        """
        Function that returns a list of current positions and a list of current velocities of all the joints of the robot
        """
        positions = []
        velocities = []

        for joint in self._motor_joints:
            joint_state = pybullet.getJointState(self._robot_id, joint.id)
            positions.append(joint_state[0])
            velocities.append(joint_state[1])

        return positions, velocities

    def _update_history(self):
        hist = []
        pos, vel = self._get_joint_states()
        hist.extend(pos)
        hist.extend(vel)

        self._joint_history.appendleft(hist)
    
    def _get_ground_feet_contacts(self) -> set:
        """
        This method saves which feet are touching the ground
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
                if link_idx -1 == joint.link_id and joint.type == "foot": # linkd indices in contacts are shifted by 1 compared to the ones stored in the joint objects (it's conventional). We apply the -1 shift to address the joint with their saved link_id
                    feet_in_contact.add(link_idx - 1)
        
        return feet_in_contact


    def _get_gravity_vector(self) -> np.ndarray:
        """
        Returns the gravity vector in the robot's base frame.
        
        Returns:
            np.ndarray: 3D vector representing gravity direction in robot base frame
        """
        # World frame gravity vector (pointing down)
        gravity_world = np.array([0, 0, -1])
        
        # Get the rotation matrix from base orientation quaternion
        base_orientation = self._agent_state["base_orientation"]
        rot_matrix = np.array(pybullet.getMatrixFromQuaternion(base_orientation)).reshape(3, 3)
        
        # Transform gravity vector from world frame to base frame
        gravity_base = rot_matrix.T @ gravity_world
        
        return gravity_base   
    
    def _get_observation(self) -> np.ndarray:
        """
        - 0-2: gravity vector
        - 3: height of the robot
        - 4-6: linear velocity of the base
        - 7-9: angular velocity of the base
        - 10-21: positions of the joints
        - 22-33: velocities of the joints
        - 34-81: history
        - 82-93: previous action
        """

        obs = []
        positions, velocities = self._get_joint_states()

        obs.extend(self._get_gravity_vector())
        obs.append((self._agent_state["base_position"])[2])
        obs.extend(self._agent_state["linear_velocity"])
        obs.extend(self._agent_state["angular_velocity"])
        obs.extend(positions)
        obs.extend(velocities)
        obs.extend(self._joint_history[1])
        obs.extend(self._joint_history[4])
        obs.extend(self._previous_action)

        assert len(obs) == self._OBS_SPACE_SIZE, f"Expected 94 elements, got {len(obs)}"

        return np.array(obs, dtype=np.float32)

    def _update_agent_state(self) -> None:
        """
        Update position, orientation and linear and angular velocities
        """

        self._agent_state["base_position"], self._agent_state["base_orientation"] = pybullet.getBasePositionAndOrientation(self._robot_id)
        self._agent_state["linear_velocity"], self._agent_state["angular_velocity"] = pybullet.getBaseVelocity(self._robot_id)
        self._agent_state["ground_feet_contacts"] = self._get_ground_feet_contacts()

        for joint in self._motor_joints:
            joint_state = pybullet.getJointState(self._robot_id, joint.id)
            joint.effort = joint_state[3]

        return  

    #@TODO: discuss the target state and rework this
    def _is_target_state(self, agent_state) -> tuple[bool, int]:
        """
        Private method that returns wether the state of the agent is a target state (one in which to end the simulation) or not. It also returns a penalty to apply when the specific target condition is met
        """

        base_pos = agent_state["base_position"]
        roll, pitch, _ = pybullet.getEulerFromQuaternion(self._agent_state["base_orientation"])
        height = base_pos[2]

        if height <= self._target_state["min_height"] or height > self._target_state["max_height"]:
            return (True, -30)
        elif abs(roll) > self._target_state["max_pitchroll"] or abs(pitch) > self._target_state["max_pitchroll"]:
            return (True, -5)
        else:
            return (False, 0)
    
    def _is_terminated(self) -> bool:
        """
        Function that returns wether an episode was terminated artificially or timed out
        """
        return (self._episode_step_counter >= self._MAX_EPISODE_LEN)

    def _get_info(self) -> dict:
        """
        Function that returns a dict containing the following fields:
            - height (of the body)
            - pitch: (of the base)
            - episode_step
        """
        return {
            "height": self._agent_state["base_position"][2],
            "pitch": pybullet.getEulerFromQuaternion(self._agent_state["base_orientation"])[1],
            "episode_step": self._episode_step_counter
        }

    def _tilt_plane(self):
        """
        Smoothly tilts the plane in one direction
        Right now it's unused
        """

        self._tilt_step += 1

        freq = 0.01
        max_angle = np.radians(6)

        tilt_x = max_angle * np.sin(freq * self._tilt_step + self._tilt_phase)
        tilt_y = max_angle * np.cos(freq * self._tilt_step + self._tilt_phase)

        quat = pybullet.getQuaternionFromEuler([tilt_x, tilt_y, 0])

        pybullet.resetBasePositionAndOrientation(
            self._plane_id,
            posObj=[0, 0, 0],
            ornObj=quat,
            physicsClientId=self.physics_client
        )

    def _calculate_reward(self, action: np.ndarray) -> tuple[float, dict]:
        """
        Placeholder method that calls the reward function provided as an input
        """
        return self._reward_fn(self, action)
