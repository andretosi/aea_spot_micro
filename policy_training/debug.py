import numpy as np
import torch
from stable_baselines3 import PPO
from SpotmicroEnv import SpotmicroEnv
from standing_reward_function import reward_function, RewardState

# ---------- CONFIG ----------
MODEL_PATH = "policies/ppo_stand"   # your pretrained file
N_STEPS = 200                  # rollout length for tests
# ----------------------------

# load model
model = PPO.load(MODEL_PATH)
model.policy.eval()

# helper: partition indices according to your doc comment in _get_observation
def partition_obs(obs):
    # using the indices you described earlier
    obs = np.asarray(obs, dtype=np.float32)
    partitions = {
        "gravity": obs[0:3],
        "height": obs[3],
        "lin_vel": obs[4:7],
        "ang_vel": obs[7:10],
        "joint_pos": obs[10:22],
        "joint_vel": obs[22:34],
        "history_1": obs[34:82],    # be careful: this is 48 long (34..81)
        "previous_action": obs[82:94] # last 12
    }
    return partitions

# Run deterministic rollout with logging of obs and actions
env = SpotmicroEnv(use_gui=False, reward_fn=reward_function, reward_state=RewardState(), dest_save_file="states/debug.pkl")
obs, info = env.reset()

# ensure obs dtype
print("obs dtype:", obs.dtype, "shape:", obs.shape)

# Print partitioned stats for initial observation
parts = partition_obs(obs)
for k,v in parts.items():
    arr = np.asarray(v)
    print(f"{k}: shape={arr.shape}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}, std={arr.std():.6f}")

# Deterministic action from policy
action, _ = model.predict(obs, deterministic=True)
print("action dtype/shape:", action.dtype, action.shape)
print("action sample:", action)

# Map actions to joint target positions (if you have a mapping function)
targets = []
for i, joint in enumerate(env._agent._motor_joints):
    targ = joint.from_action_to_position(action[i])
    targets.append(targ)
    print(f"joint[{i}] {joint.name if hasattr(joint,'name') else joint.id}: action={action[i]:.6f} -> target={targ:.6f}")

# Step the env deterministically for N_STEPS and log joint_positions, obs
joint_traj = []
obs_list = []
action_list = []
for step in range(N_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, rew, term, trunc, info = env.step(action)
    obs_list.append(obs.copy())
    action_list.append(action.copy())
    joint_traj.append(env._agent.state.joint_positions.copy())
    if term or trunc:
        print("episode ended at step", step)
        break

env.close()

# Quick summarization: variance of actions, joint_positions
action_arr = np.asarray(action_list)
joint_arr = np.asarray(joint_traj)
print("actions mean/std per-dim:", np.mean(action_arr, axis=0), np.std(action_arr, axis=0))
print("joint pos mean/std per-dim:", np.mean(joint_arr, axis=0), np.std(joint_arr, axis=0))
