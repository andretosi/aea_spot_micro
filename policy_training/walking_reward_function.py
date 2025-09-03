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
        for i in range(pybullet.getNumJoints(env._agent.agent_id)):
            joint_info = pybullet.getJointInfo(env._agent.agent_id, i)
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

def foot_clearance_reward(env, clearance_threshold=0.02):
    """
    Compute structured clearance reward relative to terrain height:
    - +1.0 if exactly one foot clears threshold
    - +0.25 if two feet clear
    - -0.5 if three or more clear
    """
    clearance_counts = 0

    for foot_id in env.reward_state.foot_ids:
        # get foot world position
        foot_pos = pybullet.getLinkState(env.agent.agent_id, foot_id)[0]
        foot_x, foot_y, foot_z = foot_pos

        # ray test to find ground under foot
        ray_start = [foot_x, foot_y, foot_z + 0.2]
        ray_end   = [foot_x, foot_y, foot_z - 0.3]
        hit = pybullet.rayTest(ray_start, ray_end, physicsClientId=env.physics_client)[0]

        if hit[0] == env.terrain.terrain_id:   # hit terrain
            terrain_height = hit[3][2]
            clearance = foot_z - terrain_height

            if clearance > clearance_threshold:
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

    roll, pitch, _ = env.agent.state.roll_pitch_yaw

    # Errors
    lin_vel_error = np.linalg.norm(env.target_lin_velocity - env.agent.state.linear_velocity) ** 2
    ang_vel_error = np.linalg.norm(env.target_ang_velocity - env.agent.state.angular_velocity) ** 2
    deviation_penalty = np.linalg.norm(env.agent.state.joint_positions - env.agent.homing_positions) ** 2
    height_penalty = (env.agent.state.base_position[2] - env.config.target_height) ** 2
    action_rate = np.linalg.norm(action - env.agent.previous_action) ** 2
    vertical_velocity_sq =  env.agent.state.linear_velocity[2] ** 2
    stabilization_penalty = roll ** 2 + pitch ** 2
    clearance_reward = foot_clearance_reward(env)

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
        "action_rate_penalty": -0.5 * action_rate,
        "deviation_penalty": -0.5 * deviation_penalty,
    }
    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict
