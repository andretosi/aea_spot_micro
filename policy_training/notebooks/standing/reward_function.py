import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

class RewardState:
    def __init__(self):
        self.prev_contacts = set()
    
    def populate(self, env: SpotmicroEnv):
        return

def fade_in(current_step, start=300_000, scale=2.0):
    if current_step < start:
        return 0.0
    return 1.0 - np.exp(-scale * (current_step - start) / 1_000_000)

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:

    roll, pitch, _ = env.agent.state.roll_pitch_yaw
    base_height = env.agent.state.base_orientation[2]
    positions = env.agent.state.joint_positions
    homing_positions = env.agent.homing_positions

    # === Uprightness Reward ===
    max_angle = np.radians(45)  # anything over this is considered tipping
    uprightness = 1.0 - (abs(roll) + abs(pitch-env.agent.config.homing_pitch)) / max_angle
    uprightness = np.clip(uprightness, 0.0, 1.0)

    # === Height Reward ===
    height_error = abs(base_height - env.config.target_height)
    height_reward = np.exp(-5 * height_error)  # 1 if perfect, drops off quickly

    # == Vertical velocity ==
    vertical_velocity_penalty = env.agent.state.linear_velocity[2] ** 2
  

    # == NEW ADDITIONS TO STABILIZE ==
    action_magnitude = np.mean(np.abs(action))
    action_sparsity_reward =  np.exp(-4 * action_magnitude) # Reward for very small actions.
    joint_deviation = np.mean(np.abs(positions - homing_positions))


    # === Final Reward ===
    reward_dict = {
        "uprightness": 2 * uprightness,
        "height": 3 * height_reward,
        "vertical_velocity_penalty": -2 * vertical_velocity_penalty,
        "joint_deviation_penalty": -4 * joint_deviation,
        "action_sparsity_reward": 1 * action_sparsity_reward,
    }
    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict
