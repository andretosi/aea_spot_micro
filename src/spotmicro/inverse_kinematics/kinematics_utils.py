import numpy as np
import pybullet as p


def extract_movable_joints_ids(body_id):
    """Extract the indexes of the movable joints (revolute and prismatic) of a body

    Args:
        body_id (int): ID of the body

    Returns:
        list[int]: List of indexes of the movable joints
    """
    num_joints = p.getNumJoints(body_id)

    movable_joints_ids = []
    for j in range(num_joints):
        if p.getJointInfo(body_id, j)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            movable_joints_ids.append(j)

    return movable_joints_ids

def reset_robot(body_id,
                joints_init_positions,
                start_position=[0, 0, 1],
                start_orientation=[0, 0, 0, 1]):
    """Reset the robot and its joints to a given position

    Args:
        body_id (int): ID of the robot body
        joints_init_positions (list): Initial positions of the movable joints
        start_position (list, optional): Initial position of the robot. Default to [0, 0, 1].
        start_orientation (list, optional): Initial quaternion orientation of the robot. Default to [0, 0, 0, 1].
    """

    movable_joints_ids = extract_movable_joints_ids(body_id)

    p.resetBasePositionAndOrientation(body_id, start_position, start_orientation)
    p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0])

    for joint_id, angle in zip(movable_joints_ids, joints_init_positions):
        p.resetJointState(body_id, joint_id, angle)

    p.setJointMotorControlArray(
        body_id,
        movable_joints_ids,
        p.POSITION_CONTROL,
        joints_init_positions
    )

def body_to_world(body_id, local_position):
    """transforms a position from the body frame to the world frame

    Args:
        body_id (int): ID of the body
        local_position (vec3): position of the point in the body frame
    
    Returns:
        vec3: position of the point in the world frame
    """

    base_pos, base_orn = p.getBasePositionAndOrientation(body_id)

    world_position, _ = p.multiplyTransforms(
        base_pos,
        base_orn,
        local_position,
        [0, 0, 0, 1]
    )

    return world_position

def world_to_body(body_id, world_position):
    """transforms a position from the world frame to the body frame

    Args:
        body_id (int): ID of the body
        world_position (vec3): position of the point in the world frame

    Returns:
        vec3: position of the point in the body frame
    """

    base_pos, base_orn = p.getBasePositionAndOrientation(body_id)
    inv_pos, inv_orn = p.invertTransform(base_pos, base_orn)

    local_position, _ = p.multiplyTransforms(
        inv_pos,
        inv_orn,
        world_position,
        [0, 0, 0, 1]
    )

    return local_position

def move_end_effectors(body_id, end_eff_ids, end_eff_targets):
    """calculate kinematics and move the end effectors to the target positions

    Args:
        body_id (int): ID of the body
        end_eff_ids (list[int]): end effectors link indexes
        end_eff_targets (list[vec3]): list of target positions for the end effectors in the world frame
    """

    solution = p.calculateInverseKinematics2(body_id, end_eff_ids, end_eff_targets)

    movable_joints_ids = extract_movable_joints_ids(body_id)

    p.setJointMotorControlArray(
        body_id,
        movable_joints_ids,
        p.POSITION_CONTROL,
        solution
    )

def walking_trajectory_local(foot_base_local,
                             walk_phase,
                             step_length,
                             step_height,
                             stance_factor,
                             forward_sign=-1):
    """calculate the  trajectory of a foot during walking in the body frame

    Args:
        foot_base_local (vec3): base position of the foot in the body frame
        walk_phase (float): walking phase [0, 1]
        step_length (float): length of each step
        step_height (float): max height of the foot during swing phase
        stance_factor (float): percentage of the step period that is spent in the stance phase (0, 1)
        forward_sign (int, optional): sign to correct walking direction in case the robot is facing the wrong way. Default to -1.

    Returns:
        vec3: target position of the foot in the body frame at the given phase of the walking cycle
    """

    x0, y0, z0 = foot_base_local

    # Robot in URDF file looks towards -x ¯\_(ツ)_/¯
    # Introducing the sign to correct walking direction
    L = forward_sign * step_length

    if walk_phase < stance_factor:
        # Stance: foot on the ground, moving backwards relative to the body
        s =  walk_phase / stance_factor
        x = x0 + (L / 2) - (s * L)
        z = z0

    else:
        # Swing: foot in the air, moving forward relative to the body
        s = (walk_phase - stance_factor) / (1 - stance_factor)
        x = x0 - L / 2 + L * (s - np.sin(2 * np.pi * s) / (2 * np.pi))
        z = z0 + step_height * (1 - np.cos(2 * np.pi * s)) / 2
    
    return [x, y0, z]
