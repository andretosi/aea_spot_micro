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
    contacts = env.agent_ground_feet_contacts
    positions, velocities = env.agent_joint_state
    homing_positions = np.array([joint.homing_position for joint in env.motor_joints])

    # === Uprightness Reward ===
    max_angle = np.radians(45)  # anything over this is considered tipping
    uprightness = 1.0 - (abs(roll) + abs(pitch-env.HOMING_PITCH)) / max_angle
    uprightness = np.clip(uprightness, 0.0, 1.0)

    # === Height Reward ===
    TARGET_HEIGHT = env.TARGET_HEIGHT  # Should be your standing height (~0.2–0.25m usually)
    height_error = abs(base_height - TARGET_HEIGHT)
    height_reward = np.exp(-10 * height_error)  # 1 if perfect, drops off quickly

    # === Foot Contact Bonus ===
    num_feet_on_ground = len(contacts)
    if num_feet_on_ground >= 3:
        contact_bonus = 1.0
    elif num_feet_on_ground == 2:
        contact_bonus = 0.5
    else:
        contact_bonus = -0.5  # unstable or collapsed

    prev_contacts = env.get_custom_state("prev_contacts")
    env.set_custom_state("prev_contacts", contacts)
    foot_stability_bonus = 0.5 if prev_contacts == contacts else 0

    # == EFFORT ==
    effort = sum(abs(joint.effort) / joint.max_torque for joint in env.motor_joints) / len(env.motor_joints)
    
    # == Joint velocity ==
    _, joint_velocities = env.agent_joint_state
    avg_joint_vel = np.mean(np.abs(joint_velocities))
    joint_velocity_penalty = np.abs(np.tanh(avg_joint_vel)) # tra 0 e 1

    # == Vertical velocity ==
    vertical_velocity_penalty = env.agent_linear_velocity[2] ** 2

    # == delta action ==
    delta_action = np.mean(np.abs(action - env.agent_previous_action))
    smoothness_penalty = delta_action**2   

    # == NEW ADDITIONS TO STABILIZE ==
    action_magnitude = np.mean(np.abs(action))
    action_sparsity_reward =  np.exp(-4 * action_magnitude) # Reward for very small actions.
    joint_deviation = np.mean(np.abs(positions - homing_positions))


    # === Final Reward ===
    reward_dict = {
        "uprightness": 1.5 * uprightness,
        "height": 2 * height_reward,
        #"contact_bonus": 1.5 * contact_bonus,
        #"stand_bonus": 1.0 if uprightness > 0.9 and height_reward > 0.9 and num_feet_on_ground >= 3 else 0.0,
        "effort_penalty": -1 * fade_in(env.num_steps, scale=2) * effort,
        "joint_velocity_penalty": -2 * joint_velocity_penalty,
        "vertical_velocity_penalty": -1.5 * vertical_velocity_penalty,
        "smoothness_penalty": -0.5 * smoothness_penalty,
        "joint_deviation_penalty": -1.5 * joint_deviation,
        "foot_stability_bonus": 2 * foot_stability_bonus,
        "action_sparsity_reward": 1 * action_sparsity_reward,
    }
    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict
