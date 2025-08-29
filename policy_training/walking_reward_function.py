import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

class RewardState:
    def __init__(self):
        self.prev_contacts = set()
        self.foot_ids = [] # Needs to be populated

    # Populate the reward state with attributes that need a ready environment to gather info
    def populate(self, env: SpotmicroEnv):
        # Init foot IDs once
        foot_ids = []
        for i in range(pybullet.getNumJoints(env._robot_id)):
            joint_info = pybullet.getJointInfo(env._robot_id, i)
            link_name = joint_info[12].decode("utf-8")
            if "foot" in link_name.lower():
                foot_ids.append(i)
        self.foot_ids = foot_ids

def fade_in(current_step, start, scale=2.0):
    if current_step < start:
        return 0.0
    return 1.0 - np.exp(-scale * (current_step - start) / 1_000_000)

def fade_out(current_step, start, scale=2.0):
    if (current_step < start):
        return 1.0
    return np.exp(-scale * (current_step - start) / 1_000_000)

def foot_clearance_reward(clearances, clearance_threshold=0.02):
    """
    Compute a structured clearance reward:
    - +1 if exactly one foot has clearance
    - +0.5 if two feet have clearance
    - -1 if three or more feet have clearance
    """
    clearance_counts = 0
    for clearance in clearances:
        if clearance > clearance_threshold: #NEED TO ADJUST FOR ROUGH TERRAIN WITH BUMPS
            clearance_counts += 1

    if clearance_counts == 1:
        return 1.0
    elif clearance_counts == 2:
        return 0.25
    elif clearance_counts >= 3:
        return -0.5
    else:
        return 0.0

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:

    positions, _ = env.agent_joint_state
    roll, pitch, _ = pybullet.getEulerFromQuaternion(env.agent_base_orientation)
    foot_positions = [pybullet.getLinkState(env._robot_id, fid)[0] for fid in env.reward_state.foot_ids]
    clearances = [pos[2] for pos in foot_positions]

    # Errors
    lin_vel_error = np.linalg.norm(env.target_lin_velocity - env.agent_linear_velocity) ** 2
    ang_vel_error = np.linalg.norm(env.target_ang_velocity - env.agent_angular_velocity) ** 2
    deviation_penalty = np.linalg.norm(positions - np.array(env.homing_positions)) ** 2
    height_penalty = (env.agent_base_position[2] - env.config.target_height) ** 2
    action_rate = np.linalg.norm(action - env.agent_previous_action) ** 2
    vertical_velocity_sq =  env.agent_linear_velocity[2] ** 2
    stabilization_penalty = roll ** 2 + pitch ** 2
    clearance_reward = foot_clearance_reward(clearances)

    # Derived penalties
    lin_vel_reward = max(1 - 1.75 * lin_vel_error, -1.0)

    # === Final Reward ===
    reward_dict = {
        "linear_vel_reward": 10 * lin_vel_reward,
        "foot_clearance_reward": 4 * clearance_reward,
        "height_penalty": -3 * min(height_penalty, 1.0),
        "stabilization_penalty": -3 * min(stabilization_penalty, 1.0),
        #"vertical_velocity_penalty": -1.5 * vertical_velocity_sq,
        "angular_vel_penalty": -1 * ang_vel_error,
        #"action_rate_penalty": -1 * action_rate,
        "deviation_penalty": -0.5 * deviation_penalty,
    }
    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict
