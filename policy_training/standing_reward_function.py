import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

def init_custom_state(env: SpotmicroEnv) -> None:
    """
    Initialize custom state tracking if needed (not much used here).
    """
    env.set_custom_state("prev_contacts", set())

def fade_in(current_step, start=300_000, scale=2.0):
    if current_step < start:
        return 0.0
    return 1.0 - np.exp(-scale * (current_step - start) / 1_000_000)

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:

    roll, pitch, _ = pybullet.getEulerFromQuaternion(env.agent_base_orientation)
    base_height = env.agent_base_position[2]
    positions, velocities = env.agent_joint_state
    homing_positions = np.array([joint.homing_position for joint in env.motor_joints])

    # === Uprightness Reward ===
    max_angle = np.radians(45)  # anything over this is considered tipping
    uprightness = 1.0 - (abs(roll) + abs(pitch-env.HOMING_PITCH)) / max_angle
    uprightness = np.clip(uprightness, 0.0, 1.0)

    # === Height Reward ===
    height_error = abs(base_height - env.config.target_height)
    height_reward = np.exp(-5 * height_error)  # 1 if perfect, drops off quickly

    # == Vertical velocity ==
    vertical_velocity_penalty = env.agent_linear_velocity[2] ** 2
  

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
