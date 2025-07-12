import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

def init_custom_state(env: SpotmicroEnv) -> None:
    """
    Initialize custom defined state variables for the environment.
    """
    env.set_custom_state("previous_gfc", set())
    env.set_custom_state("cumulative_forward_distance_bonus", 0)
    env.set_custom_state("prev_ang_vel", 0)
    env.set_custom_state("prev_lin_vel", 0)
    env.set_custom_state("prev_energy_penalty", 0)
    env.set_custom_state("effort_ema", 0)
    env.set_custom_state("symmetry_penalty_ema", 0)

    front_legs_ids = []
    back_legs_ids = []
    for i, joint in enumerate(env.motor_joints):
        if joint.name.startswith("front"):
            front_legs_ids.append(i)
        elif joint.name.startswith("rear"):
            back_legs_ids.append(i)

    env.set_custom_state("front_legs_ids", front_legs_ids)
    env.set_custom_state("back_legs_ids", back_legs_ids)
    

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:
    roll, pitch, _ = pybullet.getEulerFromQuaternion(env.agent_base_orientation)
    base_height = env.agent_base_position[2]
    linear_vel = np.array(env.agent_linear_velocity)
    angular_vel = np.array(env.agent_angular_velocity)
    contacts = env.agent_ground_feet_contacts

    fade_in_at = lambda start, scale_coefficient: 0 if env.num_steps < start else 1 - np.exp(- scale_coefficient * ((env.num_steps - start) / 1_000_000)) #0->1
    fade_out_at = lambda start, scale_coefficient: 1 if env.num_steps < start else np.exp(- scale_coefficient * ((env.num_steps - start) / 1_000_000)) #1->0
    ema = lambda alpha, curr, prev_ema: alpha * curr + (1- alpha) * prev_ema

    # === 0. Effort ===
    alpha = 0.2

    effort = 0.0
    for joint in env.motor_joints:
        effort += abs(joint.effort) / joint.max_torque #normalzing each contribution by max effort
    effort /= len(env.motor_joints) #normalizing by number of joints
    effort_ema = ema(alpha, effort, env.get_custom_state("effort_ema"))
    env.set_custom_state("effort_ema", effort_ema)
    
    front_legs_effort = 0
    back_legs_effort = 0
    for i, j in zip(env.get_custom_state("front_legs_ids"), env.get_custom_state("back_legs_ids")):
        front_legs_effort += env.motor_joints[i].effort / env.motor_joints[i].max_torque
        back_legs_effort += env.motor_joints[j].effort / env.motor_joints[j].max_torque
    symmetry_penalty = np.abs((front_legs_effort - back_legs_effort) / (len(env.motor_joints) / 2))
    symmetry_penalty_ema = ema(alpha, symmetry_penalty, env.get_custom_state("symmetry_penalty_ema"))
    env.set_custom_state("symmetry_penalty_ema", symmetry_penalty_ema) 

    # === 1. Forward Progress ===
    fwd_velocity = np.dot(linear_vel, env.target_direction)
    fwd_reward = np.clip(fwd_velocity, -1, 1)  # m/s, clip for robustness
    deviation_velocity = abs(np.dot(linear_vel, np.array([0,1,0])))
    deviation_penalty = np.clip(deviation_velocity, 0, 1)
    stillness_reward = 1.0 if np.linalg.norm(linear_vel) < 0.05 else 0.0

    vertical_velocity = abs(env.agent_linear_velocity[2])

    linear_accel = linear_vel - env.get_custom_state("prev_lin_vel")
    angular_accel = angular_vel - env.get_custom_state("prev_ang_vel")
    env.set_custom_state("prev_lin_vel", linear_vel)
    env.set_custom_state("prev_ang_vel", angular_vel)
    acceleration_penalty = 0.75 * np.clip(np.linalg.norm(linear_accel), 0.0, 1.0) + 0.25 * np.clip(np.linalg.norm(angular_accel), 0.0, 1.0)

    #== Early velocity penalty ==
    linear_vel_penalty = np.linalg.norm(env.agent_linear_velocity)  # Penalize body movement
    angular_vel_penalty = np.linalg.norm(env.agent_angular_velocity)
    _, joint_velocities = env.agent_joint_state
    joint_vel_penalty = np.mean(np.abs(joint_velocities))

    movement_penalty = 0.5 * linear_vel_penalty + 0.5 * angular_vel_penalty + 0.1 * joint_vel_penalty

    # === 2. Uprightness (Pitch & Roll) ===
    max_angle = np.radians(45)
    upright_penalty = (abs(roll) + abs(pitch)) / max_angle
    upright_reward = np.clip(1.0 - upright_penalty, 0.0, 1.0)

    # === 3. Height regulation ===
    height_target = env.TARGET_HEIGHT
    height_error = abs(base_height - height_target)
    height_reward = np.exp(-15 * height_error)

    # === 4. Energy / Smoothness ===
    tmp = np.linalg.norm(action - env.agent_previous_action) / (len(action) / 2) #Normalizing, since actions are in range -1,1
    if tmp != 0:
        energy_penalty = tmp
        env.set_custom_state("prev_energy_penalty", tmp)
    else:
        energy_penalty = env.get_custom_state("prev_energy_penalty")

    # === 5. Contact ===
    contact_bonus = 0.0
    if len(env.get_custom_state("previous_gfc")) >= 3:
        contact_bonus += 0.25
    if len(contacts) >= 3:
        contact_bonus += 1
    elif len(contacts) == 2:
        contact_bonus += 0.5
    else:
        contact_bonus -= 0.5

    stand_bonus = 1.0 if upright_reward > 0.95 and len(contacts) >= 3 else -1
    
    distance_bonus = 0
    if env.agent_base_position[0] > env.get_custom_state("cumulative_forward_distance_bonus"):
        distance_bonus += 1
        env.set_custom_state("cumulative_forward_distance_bonus", env.agent_base_position[0])
    #env.set_custom_state("previous_gfc", env.agent_ground_feet_contacts)
    #distance_penalty = np.linalg.norm(np.array([0, 0, base_height]) - env.agent_base_position)

    weights_dict = { #Doesn't work. start penalyzing actions, motion in general.
        "height": 5,
        "uprightness": 5,
        "stand_bonus": 4,
        "contact_bonus": 4,
        "movement_penalty": 0,
        "fwd_reward": 0,
        "acceleration_penalty": 0, 
        "deviation_penalty": 0, 
        "energy_penalty": 0, 
        "effort_penalty": 0,
        "total_distance_bonus": 0, 
        "symmetry_penalty": 0,
        "vertical_penalty": 0
    }

    #=== Reward weighting ===
    reward_dict = {
        "height": height_reward,
        "uprightness": upright_reward,
        "stand_bonus": stand_bonus,
        "contact_bonus": contact_bonus,
        "movement_penalty": movement_penalty,
        "fwd_reward": fwd_reward,
        "acceleration_penalty": acceleration_penalty,
        "deviation_penalty": deviation_penalty,
        "energy_penalty": energy_penalty,
        "effort_penalty": effort_ema,
        "total_distance_bonus": distance_bonus,
        "symmetry_penalty": symmetry_penalty_ema,
        "vertical_penalty": vertical_velocity
    }

    for k in reward_dict.keys():
        if k in weights_dict:
            reward_dict[k] *= weights_dict[k]
    env.log_rewards(reward_dict=reward_dict)
    total_reward = sum(reward_dict.values())
    return total_reward, reward_dict