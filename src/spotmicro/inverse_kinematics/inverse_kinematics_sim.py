import time
import os

import pybullet as p
import pybullet_data
import numpy as np

from kinematics_utils import (
    reset_robot,
    world_to_body,
    body_to_world, 
    extract_movable_joints_ids,
    walking_trajectory_local,
    move_end_effectors,
)



# MODES
# Boolean - the body is fixed in the air
SUSPENDED = False
#
# Boolean - joints stay in the position you drag them into
MODEL_MODE = False

# VARIABLES
STEP_PERIOD = 0.35
STEP_LENGTH = 0.09
STEP_HEIGHT = 0.04
STANCE_FACTOR = 0.8 # percentage of the step period that is spent in the stance phase (when the foot is on the ground)
DT = 1./240.


# FUNCTIONS

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

def reset_camera(body_id):
    """reset the camera to a default position and orientation focusing on the given body

    Args:
        body_id (int): ID of the body
    """
    base_position, _ = p.getBasePositionAndOrientation(body_id)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=0,
        cameraPitch=-10,
        cameraTargetPosition=base_position
    )



def main():
    physics_client_id = p.connect(p.GUI)

    try:
        # Setting up the simulation environment
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")

        # Uploading the robot
        urdf_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/spotmicroai.urdf")
        print(f"URDF file: {urdf_file_path}")
        start_pos = [0, 0, 0.26]
        start_orn = p.getQuaternionFromEuler([0, 0, np.pi])

        if MODEL_MODE or SUSPENDED:
            spot_id = p.loadURDF(urdf_file_path, start_pos, start_orn, useFixedBase=True)
        else:
            spot_id = p.loadURDF(urdf_file_path, start_pos, start_orn)

        num_joints = p.getNumJoints(spot_id)
        rev_joints_ids = extract_movable_joints_ids(spot_id)

        # Joints rest position
        leg_joint_rest_pos = -np.pi / 4
        foot_joint_rest_pos = np.pi / 2
        joints_rest_pos = [0, leg_joint_rest_pos, foot_joint_rest_pos] * 4

        # Toes indexes
        fl_toe_id = [i for i in range(num_joints) if p.getJointInfo(spot_id, i)[12].decode('utf-8') == 'front_left_toe_link'][0]  # 6
        fr_toe_id = [i for i in range(num_joints) if p.getJointInfo(spot_id, i)[12].decode('utf-8') == 'front_right_toe_link'][0] # 11
        rl_toe_id = [i for i in range(num_joints) if p.getJointInfo(spot_id, i)[12].decode('utf-8') == 'rear_left_toe_link'][0]   # 16
        rr_toe_id = [i for i in range(num_joints) if p.getJointInfo(spot_id, i)[12].decode('utf-8') == 'rear_right_toe_link'][0]  # 21
        toes_ids = [fl_toe_id, fr_toe_id, rl_toe_id, rr_toe_id]

        # Getting the toes rest positions in the body frame
        reset_robot(spot_id, joints_rest_pos, start_pos, start_orn) # Positioning the robot in the rest position to get the correct toes rest positions
        fl_toe_rest_pos = world_to_body(spot_id, p.getLinkState(spot_id, fl_toe_id)[0])
        fr_toe_rest_pos = world_to_body(spot_id, p.getLinkState(spot_id, fr_toe_id)[0])
        rl_toe_rest_pos = world_to_body(spot_id, p.getLinkState(spot_id, rl_toe_id)[0])
        rr_toe_rest_pos = world_to_body(spot_id, p.getLinkState(spot_id, rr_toe_id)[0])

        # !NOT USED YET! Grouping all rev_joints_ids in their own leg -> [shoulder, leg, foot]
        fl_joints_ids = [i for i in rev_joints_ids if 'front_left_' in p.getJointInfo(spot_id, i)[12].decode('utf-8')]
        fr_joints_ids = [i for i in rev_joints_ids if 'front_right_' in p.getJointInfo(spot_id, i)[12].decode('utf-8')]
        rl_joints_ids = [i for i in rev_joints_ids if 'rear_left_' in p.getJointInfo(spot_id, i)[12].decode('utf-8')]
        rr_joints_ids = [i for i in rev_joints_ids if 'rear_right_' in p.getJointInfo(spot_id, i)[12].decode('utf-8')]


        ##########################  Initial operations and simulation loop  ##########################

        reset_robot(spot_id, joints_rest_pos, start_pos, start_orn)
        reset_camera(spot_id)
        currently_walking = False
        phase = 0.

        if not MODEL_MODE:
            print_instructions()
        else:
            print_instructions_model_mode()

        while p.isConnected(physics_client_id):
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
                reset_robot(spot_id, joints_rest_pos, start_pos, start_orn)
                if not MODEL_MODE:
                    reset_camera(spot_id)
                currently_walking = False
                phase = 0.
                
            if currently_walking:
                # Walking animation

                # Increment the phase by a step. The modulo makes it loop between 0 and 1.
                phase = (phase + DT / STEP_PERIOD) % 1.

                # Calculate the target position of each foot in the body frame, then transform it to the world frame.
                # Legs on one diagonal are out of phase with the legs on the other diagonal.
                fl_target = body_to_world(spot_id, walking_trajectory_local(fl_toe_rest_pos, phase, STEP_LENGTH, STEP_HEIGHT, STANCE_FACTOR))
                fr_target = body_to_world(spot_id, walking_trajectory_local(fr_toe_rest_pos, (phase + 0.5) % 1., STEP_LENGTH, STEP_HEIGHT, STANCE_FACTOR))
                rl_target = body_to_world(spot_id, walking_trajectory_local(rl_toe_rest_pos, (phase + 0.5) % 1., STEP_LENGTH, STEP_HEIGHT, STANCE_FACTOR))
                rr_target = body_to_world(spot_id, walking_trajectory_local(rr_toe_rest_pos, phase, STEP_LENGTH, STEP_HEIGHT, STANCE_FACTOR))

                move_end_effectors(spot_id, toes_ids, [fl_target, fr_target, rl_target, rr_target])
            

            # Model mode, keep the joints in the position you drag them into
            if MODEL_MODE:
                current_pos = [j_state[0] for j_state in p.getJointStates(spot_id, rev_joints_ids)]
                p.setJointMotorControlArray(spot_id, rev_joints_ids, p.POSITION_CONTROL, current_pos)

            time.sleep(DT)
    finally:
        if p.isConnected(physics_client_id):
            p.disconnect(physics_client_id)



if __name__ == "__main__":
    main()