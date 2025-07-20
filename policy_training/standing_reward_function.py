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
    effort_penalty = fade_in(env.num_steps) * effort * 0.5
    

    # == Joint velocity ==
    _, joint_velocities = env.agent_joint_state
    avg_joint_vel = np.mean(np.abs(joint_velocities))
    velocity_penalty = np.clip(avg_joint_vel, 0, 1)

    # == delta action ==
    delta_action = np.mean(np.abs(action - env.agent_previous_action))
    smoothness_penalty = delta_action**2   

    # === Final Reward ===
    reward_dict = {
        "uprightness": uprightness,
        "height": height_reward,
        "contact_bonus": contact_bonus,
        "effort_penalty": -effort_penalty,
        "stand_bonus": 1.0 if uprightness > 0.9 and height_reward > 0.9 and num_feet_on_ground >= 3 else 0.0,
        "velocity_penalty": velocity_penalty * 2,
        "smoothness_penalty": delta_action,
        "foot_stability_bonus": foot_stability_bonus
    }
    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict
