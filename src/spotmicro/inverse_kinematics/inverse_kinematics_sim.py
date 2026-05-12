import pybullet as p
import pybullet_data
import numpy as np
import time

# MODES
# Boolean - the body is fixed in the air
suspended = 0
#
# Boolean - joints stay in the position you drag them into
model_mode = 0

# VARIABLES
step_period = 0.35
step_length = 0.09
step_height = 0.04
stance_factor = 0.8 # percentage of the step period that is spent in the stance phase (when the foot is on the ground)
dt = 1./240.

# Initialize environment
physics_client_id = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")

# Uploading the robot
start_pos = [0, 0, 0.26]
start_orientation = p.getQuaternionFromEuler([0, 0, np.pi])

if model_mode or suspended:
    spot_id = p.loadURDF("spotmicroai.urdf", start_pos, start_orientation, useFixedBase=True)
else:
    spot_id = p.loadURDF("spotmicroai.urdf", start_pos, start_orientation)

# Extracting revolution joints
num_joints = p.getNumJoints(spot_id)

rev_joints_ids = []
for j in range(num_joints):
    if p.getJointInfo(spot_id, j)[2] == p.JOINT_REVOLUTE:
        rev_joints_ids.append(j)

num_rev_joints = len(rev_joints_ids)

# Joints rest position
leg_joint_rest_pos = -np.pi / 4
foot_joint_rest_pos = np.pi / 2
joints_rest_pos = [0, leg_joint_rest_pos, foot_joint_rest_pos] * 4

def reset_robot():
    """reset the robot to the initial position, which has the joint angles defined in joints_rest_pos
    """
    p.resetBasePositionAndOrientation(spot_id, start_pos, start_orientation)
    p.resetBaseVelocity(spot_id, [0, 0, 0], [0, 0, 0])

    for joint_id, angle in zip(rev_joints_ids, joints_rest_pos):
        p.resetJointState(spot_id, joint_id, angle)

    p.setJointMotorControlArray(
        spot_id,
        rev_joints_ids,
        p.POSITION_CONTROL,
        joints_rest_pos
    )

# Frame transformations
def body_to_world(local_pos):
    """transforms a position from the body frame to the world frame

    Args:
        local_pos (vec3): position of the point in the body frame
    
    Returns:
        vec3: position of the point in the world frame
    """

    base_pos, base_orn = p.getBasePositionAndOrientation(spot_id)

    world_pos, _ = p.multiplyTransforms(
        base_pos,
        base_orn,
        local_pos,
        [0, 0, 0, 1]
    )

    return world_pos

def world_to_body(world_pos):
    """transforms a position from the world frame to the body frame

    Args:
        world_pos (vec3): position of the point in the world frame

    Returns:
        vec3: position of the point in the body frame
    """

    base_pos, base_orn = p.getBasePositionAndOrientation(spot_id)
    inv_pos, inv_orn = p.invertTransform(base_pos, base_orn)

    local_pos, _ = p.multiplyTransforms(
        inv_pos,
        inv_orn,
        world_pos,
        [0, 0, 0, 1]
    )

    return local_pos

# Toes ids and rest positions
fl_toe_id = [i for i in range(num_joints) if p.getJointInfo(spot_id, i)[12].decode('utf-8') == 'front_left_toe_link'][0]  # 6
fr_toe_id = [i for i in range(num_joints) if p.getJointInfo(spot_id, i)[12].decode('utf-8') == 'front_right_toe_link'][0] # 11
rl_toe_id = [i for i in range(num_joints) if p.getJointInfo(spot_id, i)[12].decode('utf-8') == 'rear_left_toe_link'][0]   # 16
rr_toe_id = [i for i in range(num_joints) if p.getJointInfo(spot_id, i)[12].decode('utf-8') == 'rear_right_toe_link'][0]  # 21
toes_ids = [fl_toe_id, fr_toe_id, rl_toe_id, rr_toe_id]

reset_robot() # Positioning the robot in the rest position to get the correct toes rest positions
fl_toe_rest_pos = world_to_body(p.getLinkState(spot_id, fl_toe_id)[0])
fr_toe_rest_pos = world_to_body(p.getLinkState(spot_id, fr_toe_id)[0])
rl_toe_rest_pos = world_to_body(p.getLinkState(spot_id, rl_toe_id)[0])
rr_toe_rest_pos = world_to_body(p.getLinkState(spot_id, rr_toe_id)[0])

# Grouping all rev_joints_ids in their own leg -> [shoulder, leg, foot]
fl_joints_ids = [i for i in rev_joints_ids if 'front_left_' in p.getJointInfo(spot_id, i)[12].decode('utf-8')]
fr_joints_ids = [i for i in rev_joints_ids if 'front_right_' in p.getJointInfo(spot_id, i)[12].decode('utf-8')]
rl_joints_ids = [i for i in rev_joints_ids if 'rear_left_' in p.getJointInfo(spot_id, i)[12].decode('utf-8')]
rr_joints_ids = [i for i in rev_joints_ids if 'rear_right_' in p.getJointInfo(spot_id, i)[12].decode('utf-8')]

# Toes control
def move_toes(targets):
    """calculate kinematics and move the toes to the target positions

    Args:
        targets (list[vec3]): list of 4 target positions for the toes in the world frame
    """

    solution = p.calculateInverseKinematics2(spot_id, toes_ids, targets)
    p.setJointMotorControlArray(
        spot_id,
        rev_joints_ids,
        p.POSITION_CONTROL,
        solution
    )

def walking_trajectory_local(local_foot_base, phase, step_length, step_height, forward_sign=-1):
    """calculate the  trajectory of a foot during walking in the body frame

    Args:
        local_foot_base (vec3): base position of the foot in the body frame
        phase (float): walking phase [0, 1]
        step_length (float): length of each step
        step_height (float): max height of the foot during swing phase
        forward_sign (int, optional): sign to correct walking direction. Defaults to -1.

    Returns:
        vec3: target position of the foot in the body frame
    """

    x0, y0, z0 = local_foot_base

    # Robot in URDF file looks towards -x ¯\_(ツ)_/¯
    # Introducing the sign to correct walking direction
    L = forward_sign * step_length

    if phase < stance_factor:
        # Stance: foot on the ground, moving backwards relative to the body
        s =  phase / stance_factor
        x = x0 + (L / 2) - (s * L)
        z = z0

    else:
        # Swing: foot in the air, moving forward relative to the body
        s = (phase - stance_factor) / (1 - stance_factor)
        x = x0 - L / 2 + L * (s - np.sin(2 * np.pi * s) / (2 * np.pi))
        z = z0 + step_height * (1 - np.cos(2 * np.pi * s)) / 2
    
    return [x, y0, z]

# Other functions
def print_instructions():
    print("\n" * 3)
    print("-" * 50)
    print("     [Enter] to toggle walking animation.")
    print("     [R] to reset the robot position.")
    print("     [ESC] to exit.")
    print("-" * 50)

def print_instructions_model_mode():
    print("\n" * 3)
    print("-" * 70)
    print("Model mode: joints will stay in the position you drag them into.")
    print("")
    print("     [R] to reset the robot position.")
    print("     [ESC] to exit.")
    print("-" * 70)

def reset_camera():
    base_pos, _ = p.getBasePositionAndOrientation(spot_id)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=0,
        cameraPitch=-10,
        cameraTargetPosition=base_pos
    )


##########################  Initial operations and simulation loop  ##########################

reset_robot()
reset_camera()
currently_walking = False
phase = 0.

if not model_mode:
    print_instructions()
else:
    print_instructions_model_mode()

while True:
    p.stepSimulation()
    keys = p.getKeyboardEvents()

    # When ESC is pressed: exit
    if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
        break

    # When Enter is pressed: toggle walking animation
    if p.B3G_RETURN in keys and keys[p.B3G_RETURN] & p.KEY_WAS_TRIGGERED:
        currently_walking = not currently_walking

    # When R is pressed: reset robot position
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        reset_robot()
        if not model_mode:
            reset_camera()
        currently_walking = False
        phase = 0.
        
    if currently_walking:
        # Walking animation

        phase = (phase + dt / step_period) % 1.

        fl_target = body_to_world(walking_trajectory_local(fl_toe_rest_pos, phase, step_length, step_height))
        fr_target = body_to_world(walking_trajectory_local(fr_toe_rest_pos, (phase + 0.5) % 1., step_length, step_height))
        rl_target = body_to_world(walking_trajectory_local(rl_toe_rest_pos, (phase + 0.5) % 1., step_length, step_height))
        rr_target = body_to_world(walking_trajectory_local(rr_toe_rest_pos, phase, step_length, step_height))

        move_toes([fl_target, fr_target, rl_target, rr_target])
    
    # Model mode
    if model_mode:
        current_pos = [j_state[0] for j_state in p.getJointStates(spot_id, rev_joints_ids)]
        p.setJointMotorControlArray(spot_id, rev_joints_ids, p.POSITION_CONTROL, current_pos)

    time.sleep(dt)

p.disconnect()