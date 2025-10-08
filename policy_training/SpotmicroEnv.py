import pybullet, pybullet_data
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
import inspect, os, pickle, warnings
from Config import Config
from Agent import Agent
from Terrain import Terrain
"""
DOMANDE
1) perché ConfigEnv e Config class sono separate nel codice?

"""
"""
This class inherits the Config class.
First it creates and inizializes the attributes that show up in the .yaml
file given in input for the constructor.
Then it adds additional three attributes shown in the attributes array. 
"""
class ConfigEnv(Config):
    def __init__(self, filename: str):
        super().__init__(filename)
        #these attributes are used to configure the terrain
        attributes = ["c_potholes", "c_ridges", "c_roughness"]
        for attr in attributes:
            if not hasattr(self, attr):
                setattr(self, attr, 0.0)
        

"""
the SpotMicroEnv class inherits the gym.Env class and overrides his methods in order
to create an interface between Stable Baselines3 and pybullet:

SB3 <-> Env <-> Pybullet

1) ATTRIBUTES
These two attributes come from the gym.Env class
- observation_space: the Space object corresponding to valid observations
- action_space: The Space object corresponding to valid actions

These are the most important attributes that describe the SpotMicroEnv class:
_OBS_SPACE_SIZE: size of the space vector
_ACT_SPACE_SIZE: size of the action vector, returned by the neuralnet
_target_state: dictionary that defines the ranges of height and pitch/roll in whitch the robot has to stay to
               not terminate the episode. The values are taken from the envConfig.yaml file, "Target Values" section
_reward_fn:

physics_client: This object comes from the PyBullet class, and is the one who is given as input to
                most of the pyBullet methods. 

_terrain: This object belongs to the Terrain class, defined in the Terrain.py file. It contains all the info about 
          the terrain of the simulation. It takes the physics_client as input to update the ground of the simulation,
          other parameters are defined in the terrainConfig.yaml file.
_agent: 

2) METHODS
Here is a list of the methods of the gym.Env class that are overriden:
- step: Method exposed and used by SB3 to execute one time step within the environment.
    
    input:
        action (gym.spaces.Box): The action to take in the environment.

    output:
        tuple containing
            - observation (np.ndarray): Agent's observation of the environment.
            - reward (float): Amount of reward returned after previous action.
            - terminated (bool): Whether the episode naturally ended.
            - truncated (bool): Whether the episode was artificially terminated.
            - info (dict): Contains auxiliary diagnostic information. See _get_info()

    The action is taken using _step_simulation(action) method. The control loop is slowed down
    manually. 



- reset: resets the environment to an inital state, and returns the firts observation value.
    More specifically:
    Resets Agent and Terrain classes using their reset() methods; steps the pybullet
    simulation a few times until the bot is stable in the initial state. 
    Returns the initial observation and info dict, see _get_info()

- close: Closes the pybullet simulation, saves the state if a destination path is provided.

- render (ignored)

These are public methods 
- save_state: saves a dictionary containing useful data about the current state:
              (total steps counter, previous action, joint state history)
              The file is stored in _dest_save_file, parameter of the constructor method: src_save_file
- load_state: loads the dictionary saved by save_state(). The path of the file is a 
              parameter of the constructor method: dest_save_file

Private methods
- _step_simulation: accepts an action and returns an observation. This is done by the Agent class:
                    apply_action(action) -> pybullet.stepSimulation -> sync_state()
                    Eventually tilts the plane if a variable is set.
                    Returns the observation using _get_observation().

These methods are used to compute the values that are returned by step() (and other relevant methods):
- _get_observation (np.ndarray): Agent's observation of the environment.
        - 0-2: gravity vector
        - 3: height of the robot
        - 4-6: linear velocity of the base
        - 7-9: angular velocity of the base
        - 10-21: positions of the joints
        - 22-33: velocities of the joints
        - 34-81: history
        - 82-93: previous action

- _calculate_reward (float): Placeholder method that calls the reward function provided as an input.

- _is_target_state (bool): Private method that returns wether the state of the agent is a target state 
    (one in which to end the simulation) or not. It also returns a penalty to 
    apply when the specific target condition is met.

- _is_terminated (bool): Function that returns wether an episode was terminated artificially or timed out

- _get_info (dict): Function that returns a dict containing the following fields:
        - height (of the body)
        - pitch: (of the base)
        - episode_step
"""
class SpotmicroEnv(gym.Env):
    def __init__(self, envConfig="envConfig.yaml", agentConfig="agentConfig.yaml", terrainConfig="terrainConfig.yaml", use_gui=False, reward_fn=None, reward_state=None, dest_save_file=None, src_save_file=None, writer=None):
        """
        Parameters
        ------------
        - envConfig : str

        - agentConfig : str

        - terrainConfig : str

        - use_gui : bool

        - reward_fn : ?

        - reward_state : type of data?

        - dest_save_file : ?

        - src_save_file : ?

        - writer: ?
        """
        super().__init__()

        if not isinstance(envConfig, str):
            raise TypeError("config must be a string")
        if not os.path.isfile(agentConfig):
            raise FileNotFoundError(f"File {agentConfig} does not exist")
        if not os.path.isfile(terrainConfig):
            raise FileNotFoundError(f"File {terrainConfig} does not exist")

        #Config object contains only attributes, whitch value can be set frome the .yaml file
        self.config = ConfigEnv(envConfig)

        self.physics_client = None
        self.use_gui = use_gui
        self.np_random = None
        self.reward_state = reward_state
        self._episode_reward_info = None #history of the rewards during an episode, used to plot results
        self.writer = writer

        #these two constants will be assigned to the obs_space and action_space attributes
        #of the gym.Env class
        self._OBS_SPACE_SIZE = 94
        self._ACT_SPACE_SIZE = 12
        
        self._MAX_EPISODE_LEN = 3000
        self._SIM_FREQUENCY = 240
        self._CONTROL_FREQUENCY = 60
        self._JOINT_HISTORY_MAX_LEN = 5

        self._TARGET_LINEAR_VELOCITY = np.array([0.3, 0.0, 0.0])
        self._TARGET_ANGULAR_VELOCITY = np.array([0.0, 0.0, 0.0])

        self._episode_step_counter = 0
        self._total_steps_counter = 0

        #Declaration of observation and action spaces (gym.Env native attributes)
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

        #If the agents is in this state, we terminate the simulation. Should quantize the fact that it has fallen, maybe a threshold?
        self._target_state = {
            "min_height": self.config.min_height,
            "max_height": self.config.max_height,
            "max_pitchroll": self.config.max_pitchroll
        }

        if reward_fn is None:
            raise ValueError("reward_fn cannot be None. Provide a valid rewad function")
        elif not callable(reward_fn):
            raise ValueError("reward_fn must be callable (function)")

        self._reward_fn = reward_fn

        #Initialize pybullet: connects to the simulation server and sets the initial view
        if self.physics_client is None:
            self.physics_client = pybullet.connect(pybullet.GUI if self.use_gui else pybullet.DIRECT)
            pybullet.resetDebugVisualizerCamera(
                cameraDistance=1.2,   # zoom out a bit
                cameraYaw=45,         # rotate around robot
                cameraPitch=-30,      # look slightly down
                cameraTargetPosition=[0, 0, 0.2]  # center around robot base
            )

        pybullet.resetSimulation(physicsClientId=self.physics_client)
        pybullet.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        pybullet.setTimeStep(1/self._SIM_FREQUENCY, physicsClientId=self.physics_client)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        #Initialize the terrain object
        self._terrain = Terrain(self.physics_client, Config(terrainConfig))

        self._terrain_evo_coefficients = np.array([self.config.c_potholes, self.config.c_ridges, self.config.c_roughness])
        self._terrain.generate(self._terrain_evo_coefficients)
        #???
        pybullet.changeDynamics(
            bodyUniqueId=self._terrain.terrain_id,
            linkIndex=-1,
            lateralFriction=1.0,           
            spinningFriction=0.0,
            rollingFriction=0.0,
            restitution=0.0,
            physicsClientId=self.physics_client
        )

        #Initialize the agent object
        self._agent = Agent(self.physics_client, Config(agentConfig), self._ACT_SPACE_SIZE, self.config.spawn_height)

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
        
    # a cosa serve????        
    def save_state(self):
        state = {
            "total_steps_counter": self._total_steps_counter,
            "previous_action": self._agent.previous_action,
            "joint_history": list(self._agent.joint_history),
            "target_linear_velocity": self._TARGET_LINEAR_VELOCITY,
            "target_angular_velocity": self._TARGET_ANGULAR_VELOCITY
        }

        with open(self._dest_save, "wb") as f:
            pickle.dump(state, f)

    # a cosa serve????   
    def load_state(self):
        with open(self._src_file, 'rb') as f:
            state = pickle.load(f)
        
        self._total_steps_counter = state["total_steps_counter"]
        self._agent.previous_action = state["previous_action"]
        self._agent.joint_history = deque(state["joint_history"], maxlen=self._agent.config.joint_history_maxlen)
        #self._TARGET_LINEAR_VELOCITY = state["target_linear_velocity"]
        #self._TARGET_ANGULAR_VELOCITY = state["target_angular_velocity"]

    def close(self):
        """
        Method exposed and used by SB3.
        Cleans up the simulation, saves the state if a destination path is provided
        """
        if self.physics_client is not None:
            pybullet.disconnect(self.physics_client)
            self.physics_client = None

        if self._dest_save is not None:
            self.save_state() #?

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """
        Gymnasium reset: start a new episode, set counters, reset agent and terrain,
        and return the initial observation and info dict.
        """
        super().reset(seed=seed)

        self._episode_step_counter = 0
        self._episode_reward_info = []

        # Reset terrain before agent so ground height is consistent
        self._terrain.reset()
        if self._terrain.config.evolving: #evolving attribute not present??? 
            self._terrain.generate(self._schedule_terrain_evo())
        self._agent.reset(self.config.spawn_height)

        if self.reward_state is not None:
            self.reward_state.populate(self)
        else:
            print("Reward state is None")

        # la sequenza è: resetto la simulazione impostando i valori base di posizione etc....
        # poi quali azioni eseguo? a cosa serve questo apply action?

        # Let physics settle with homing applied
        for _ in range(5):
            self._agent.apply_action(self._agent.default_actions) # Zeros because they map to homing positions
            pybullet.stepSimulation(physicsClientId=self.physics_client)
            self._agent.sync_state()

        # Sanity check reward function signature
        sig = inspect.signature(self._reward_fn)
        if len(sig.parameters) != 2:
            raise ValueError("reward_fn must accept exactly 2 parameters (env, action)")

        # Test reward function return type
        try:
            dummy_action = np.array(self._agent.homing_positions, dtype=np.float32)
            reward, info = self._reward_fn(self, dummy_action)
            if not isinstance(reward, (int, float)):
                raise ValueError("reward_fn must return a number as first return value")
            if not isinstance(info, dict):
                raise ValueError("reward_fn must return a dict as second return value")
        except Exception as e:
            raise ValueError(f"Error testing reward_fn: {str(e)}")

        return self._get_observation(), self._get_info()
    
    # ????
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
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
        if (self._episode_step_counter % int(self._SIM_FREQUENCY / self._CONTROL_FREQUENCY)) == 0: # apply new action
            observation = self._step_simulation(action)
            reward, reward_info = self._calculate_reward(action)
        else:                                                                         # reuse last action
            observation = self._step_simulation(self._agent.previous_action)
            reward, reward_info = self._calculate_reward(self._agent.previous_action)

        terminated, term_penalty = self._is_target_state() # checks wether the agent has fallen or not
        truncated = self._is_terminated()
        info = self._get_info()

        #actions that are reused in the control loop still receive a reward and are appended to the episode_reward_info ????
        self._episode_reward_info.append(reward_info)
        if truncated:
            reward += self.config.survival_reward
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

    def _step_simulation(self, action: np.ndarray) -> np.ndarray:
        """
        Private method that calls the API to execute the given action in PyBullet.
        It should sinchronize the state of the agent in the simulation with the state recorded here!
        Accepts an action and returns an observation
        """
        # Execute the action in pybullet
        self._agent.apply_action(action)
        
        self._episode_step_counter += 1 #updates the step counter (used to check against timeouts)
        
        if self._terrain.config.mode == "tilting":
            self._terrain.tilt_plane()
        
        pybullet.stepSimulation()
        self._agent.sync_state()

        return self._get_observation()


    def _get_gravity_vector(self) -> np.ndarray:
        """
        Returns the gravity vector in the robot's base frame.
        
        Returns:
            np.ndarray: 3D vector representing gravity direction in robot base frame
        """
        # World frame gravity vector (pointing down)
        gravity_world = np.array([0, 0, -1])
        
        # Get the rotation matrix from base orientation quaternion
        rot_matrix = np.array(pybullet.getMatrixFromQuaternion(self._agent.state.base_orientation)).reshape(3, 3)
        
        # Transform gravity vector from world frame to base frame
        gravity_base = rot_matrix.T @ gravity_world
        
        return gravity_base   
    
    def _joint_positions_norm(self, pos):
        pos_norm = []
        for i, joint in enumerate(self._agent.motor_joints):
            pos_norm.append(((2 * (pos[i] - joint.limits[0])) / (joint.limits[1] - joint.limits[0])) - 1) # Normalize ang position with respect to max range of motion
        return pos_norm
    
    def _joint_velocities_norm(self, vels): #PARAMETER (normalization)
        vel_norm = [np.tanh(vel / self.config.max_joint_velocity) for vel in vels] # Normalize velocity with resect to a hypotetical max velocity (10 rad/s)
        return vel_norm

    
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

        #NORMALIZATION PARAMETERS
        obs = []
        obs.extend(self._get_gravity_vector())
        obs.append((self._agent.state.base_position[2] - self.config.target_height) / self.config.max_norm_height) # Normalized w respect a hypotetical max height
        obs.extend(self._agent.state.linear_velocity / self.config.max_linear_velocity) # Normalized w respect to a hypotetical max velocity
        obs.extend(self._agent.state.angular_velocity / self.config.max_angular_velocity) # Normalized w respect to a hypotetical max ang velocity
        obs.extend(self._joint_positions_norm(self._agent.state.joint_positions)) 
        obs.extend(self._joint_velocities_norm(self._agent.state.joint_velocities))
        obs.extend(self._joint_positions_norm(self._agent.joint_history[1][0]))
        obs.extend(self._joint_velocities_norm(self._agent.joint_history[1][1]))
        obs.extend(self._joint_positions_norm(self._agent.joint_history[4][0]))
        obs.extend(self._joint_velocities_norm(self._agent.joint_history[4][1]))
        obs.extend(self._agent.previous_action)

        assert len(obs) == self._OBS_SPACE_SIZE, f"Expected 94 elements, got {len(obs)}"

        return np.array(obs, dtype=np.float32)

    def _is_target_state(self) -> tuple[bool, int]:
        """
        Private method that returns wether the state of the agent is a target state (one in which to end the simulation) or not. It also returns a penalty to apply when the specific target condition is met
        """

        base_pos = self._agent.state.base_position
        roll, pitch, _ = self._agent.state.roll_pitch_yaw
        height = base_pos[2]

        if height <= self._target_state["min_height"] or height > self._target_state["max_height"]:
            return (True, self.config.jump_fall_penalty) 
        elif abs(roll) > self._target_state["max_pitchroll"] or abs(pitch) > self._target_state["max_pitchroll"]:
            return (True, self.config.tipping_penalty)
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
            "height": self._agent.state.base_position[2],
            "pitch": self._agent.state.roll_pitch_yaw[1],
            "episode_step": self._episode_step_counter
        }

    def _calculate_reward(self, action: np.ndarray) -> tuple[float, dict]:
        """
        Placeholder method that calls the reward function provided as an input
        """
        return self._reward_fn(self, action)

    def _schedule_terrain_evo(self) -> np.ndarray:
        if self.config.evolution_mode == "constant":
            return self._terrain_evo_coefficients
        elif self.config.evolution_mode == "linear":
            linear_alpha = self._total_steps_counter / 1_000_000 # TOTALLY UNSTABLE TODO NEED TO NORMALIZE
            return self._terrain_evo_coefficients * (1 - linear_alpha) + np.array([linear_alpha for coeff in self._terrain_evo_coefficients])

    
    @property
    def agent(self):
        return self._agent

    @property
    def terrain(self):
        return self._terrain
    
    @property
    def target_lin_velocity(self) -> np.ndarray:
        """
        Get the current target direction for locomotion (unit vector).
        """
        return self._TARGET_LINEAR_VELOCITY
    
    @target_lin_velocity.setter
    def target_linear_velocity(self, lin_velocity: tuple[float, float, float]) -> None:
        """
        Set a new target direction for locomotion. Should be a normalized 3D vector
        """
        norm = np.linalg.norm(lin_velocity)
        if norm == 0:
            raise ValueError("Target direction cannot be a zero vector")
        self._TARGET_LINEAR_VELOCITY = np.array(np.array(lin_velocity) / norm)
    
    @property
    def target_ang_velocity(self) -> np.ndarray:
        """
        Get the current target direction for locomotion (unit vector).
        """
        return self._TARGET_ANGULAR_VELOCITY
    
    @target_ang_velocity.setter
    def target_angular_velocity(self, ang_velocity: tuple[float, float, float]) -> None:
        """
        Set a new target direction for locomotion. Should be a normalized 3D vector
        """
        norm = np.linalg.norm(ang_velocity)
        if norm == 0:
            raise ValueError("Target direction cannot be a zero vector")
        self._TARGET_LINEAR_VELOCITY = np.array(np.array(ang_velocity) / norm)
    
    @property
    def num_steps(self) -> int:
        """
        Return the current number of steps
        """
        return self._total_steps_counter

    @property
    def sim_frequency(self) -> int:
        return self._SIM_FREQUENCY
