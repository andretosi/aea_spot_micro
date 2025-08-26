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

    roll, pitch, _ = pybullet.getEulerFromQuaternion(env.agent_base_orientation)
    base_height = env.agent_base_position[2]
    contacts = env.agent_ground_feet_contacts

    # === Uprightness Reward ===
    max_angle = np.radians(45)  # anything over this is considered tipping
    uprightness = 1.0 - (abs(roll) + abs(pitch-env.config.homing_pitch)) / max_angle
    uprightness = np.clip(uprightness, 0.0, 1.0)

    # === Height Reward ===
    height_error = abs(base_height - env.config.target_height)
    height_reward = np.exp(-10 * height_error)  # 1 if perfect, drops off quickly

    # === Foot Contact Bonus ===
    num_feet_on_ground = len(contacts)
    if num_feet_on_ground >= 3:
        contact_bonus = 1.0
    elif num_feet_on_ground == 2:
        contact_bonus = 0.5
    else:
        contact_bonus = -0.5  # unstable or collapsed

    prev_contacts = env.reward_state.prev_contacts
    env.reward_state.prev_contacts = contacts
    foot_stability_bonus = 0.5 if prev_contacts == contacts else 0

    # == EFFORT ==
    effort = sum(abs(joint.effort) / joint.max_torque for joint in env.motor_joints) / len(env.motor_joints)
    effort_penalty = effort * 0.5
    

    # == Joint velocity ==
    _, joint_velocities = env.agent_joint_state
    avg_joint_vel = np.mean(np.abs(joint_velocities))
    velocity_penalty = np.clip(avg_joint_vel, 0, 1)

    # == delta action ==
    delta_action = np.mean(np.abs(action - env.agent_previous_action))
    smoothness_penalty = delta_action**2

    # == FORWARD REWARD ==
    target_direction = env.target_direction  # Assume [1, 0, 0] for +x
    velocity = np.array(env.agent_linear_velocity)
    forward_velocity = np.dot(velocity, target_direction)
    fwd_reward = np.clip(forward_velocity, -1.0, 1.0)

    stillness_penalty = np.exp(-50 * np.linalg.norm(env.agent_linear_velocity))

    # === Final Reward ===
    reward_dict = {
        "uprightness": 1.5 * uprightness,
        "height": 2 * height_reward,
        "contact_bonus": 1.5 * contact_bonus,
        "effort_penalty": -1 * fade_in(env.num_steps, 13_000_000, 2) * effort_penalty,
        "stand_bonus": 1.0 if uprightness > 0.9 and height_reward > 0.9 and num_feet_on_ground >= 3 else 0.0,
        "velocity_penalty": -1 * fade_in(env.num_steps, 13_000_000, 2) * velocity_penalty,
        "smoothness_penalty": -1 * smoothness_penalty,
        "foot_stability_bonus": 1 * foot_stability_bonus,
        "fwd_reward": (10 + 5 * fade_out(env.num_steps, 14_000_000, 2)) * fwd_reward,
        "stillness_penalty": (-5 + 2 * fade_in(env.num_steps, 12_000_000, 1.5)) * stillness_penalty
    }
    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict
