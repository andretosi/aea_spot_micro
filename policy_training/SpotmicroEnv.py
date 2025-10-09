import pybullet, pybullet_data
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
import inspect, os, pickle, warnings
from Config import Config
from Agent import Agent
from Terrain import Terrain
import time

class ConfigEnv(Config):
    def __init__(self, filename: str):
        super().__init__(filename)

        attributes = ["c_potholes", "c_ridges", "c_roughness"]
        for attr in attributes:
            if not hasattr(self, attr):
                setattr(self, attr, 0.0)

class SpotmicroEnv(gym.Env):
    def __init__(self, envConfig="envConfig.yaml", agentConfig="agentConfig.yaml", terrainConfig="terrainConfig.yaml", use_gui=False, reward_fn=None, reward_state=None, dest_save_file=None, src_save_file=None, writer=None):
        super().__init__()

        if not isinstance(envConfig, str):
            raise TypeError("config must be a string")
        if not os.path.isfile(agentConfig):
            raise FileNotFoundError(f"File {agentConfig} does not exist")
        if not os.path.isfile(terrainConfig):
            raise FileNotFoundError(f"File {terrainConfig} does not exist")

        self.config = ConfigEnv(envConfig)

        self.physics_client = None
        self.use_gui = use_gui
        self.np_random = None
        self.reward_state = reward_state
        self._episode_reward_info = None
        self.writer = writer

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

        #Initialize pybullet
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
        
        self._terrain = Terrain(self.physics_client, Config(terrainConfig))

        self._terrain_evo_coefficients = np.array([self.config.c_potholes, self.config.c_ridges, self.config.c_roughness])
        self._terrain.generate(self._terrain_evo_coefficients)
        pybullet.changeDynamics(
            bodyUniqueId=self._terrain.terrain_id,
            linkIndex=-1,
            lateralFriction=1.0,           
            spinningFriction=0.0,
            rollingFriction=0.0,
            restitution=0.0,
            physicsClientId=self.physics_client
        )
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
            self.save_state()

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
        if self._terrain.config.evolving:
            self._terrain.generate(self._schedule_terrain_evo())
        self._agent.reset(self.config.spawn_height)

        if self.reward_state is not None:
            self.reward_state.populate(self)
        else:
            print("Reward state is None")

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
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # TODO: BUG fix here. Decouple logic steps from phisical sim steps. WOuld this loop work in a real setting? no, because what reward would you give in between the control steps? it is not feasible to calculate reward at 240Hz. So, step once giving an action and calculating the reward (on which action of the batch?), then step x steps in the sim all for one spotenv.step() call
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

        observation = self._step_simulation(action)
        self._episode_step_counter += 1 #updates the step counter (used to check against timeouts)
        reward, reward_info = self._calculate_reward(action)

        terminated, term_penalty = self._is_target_state() # checks wether the agent has fallen or not
        truncated = self._is_truncated()
        info = self._get_info()

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
        Accepts an action and returns an observation.
        The simulation is stepped multiple times to slow down the control loop
        """
        # Execute the action in pybullet
        self._agent.apply_action(action)
                
        if self._terrain.config.mode == "tilting":
            self._terrain.tilt_plane()
        
        for _ in range(self._SIM_FREQUENCY // self._CONTROL_FREQUENCY):
            pybullet.stepSimulation()
            if self.use_gui:
                time.sleep(1/70.) # MAGIC NUMBER, MAKES THE SIMULATION LOOK REAL-TIME (not slow, not too fast)

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
    
    def _is_truncated(self) -> bool:
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
