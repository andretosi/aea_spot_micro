import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

class RewardState:
    def __init__(self):
        self.prev_contacts = set()

def fade_in(current_step, start, scale=2.0):
    if current_step < start:
        return 0.0
    return 1.0 - np.exp(-scale * (current_step - start) / 1_000_000)

def fade_out(current_step, start, scale=2.0):
    if (current_step < start):
        return 1.0
    return np.exp(-scale * (current_step - start) / 1_000_000)

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:

    positions, _ = env.agent_joint_state
    roll, pitch, _ = pybullet.getEulerFromQuaternion(env.agent_base_orientation)

    lin_vel_error = np.linalg.norm(env.target_lin_velocity - env.agent_linear_velocity) ** 2
    ang_vel_error = np.linalg.norm(env.target_ang_velocity - env.agent_angular_velocity) ** 2
    deviation_penalty = np.linalg.norm(positions - np.array(env.homing_positions)) ** 2
    height_penalty = (env.agent_base_position[2] - env.config.target_height) ** 2
    action_rate = np.linalg.norm(action - env.agent_previous_action) ** 2
    vertical_velocity_sq =  env.agent_linear_velocity[2] ** 2
    stabilization_penalty = roll ** 2 + pitch ** 2

    # === Final Reward ===
    reward_dict = {
        "linear_vel_penalty": -2 * lin_vel_error,
        "height_penalty": -2 * height_penalty,
        "stabilization_penalty": -1.5 * stabilization_penalty,
        "vertical_velocity_penalty": -1.5 * vertical_velocity_sq,
        "angular_vel_penalty": -1 * ang_vel_error,
        "action_rate_penalty": -1 * action_rate,
        "deviation_penalty": -0.5 * deviation_penalty,
    }
    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict
