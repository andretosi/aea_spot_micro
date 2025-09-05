import time
import numpy as np
from stable_baselines3 import PPO

import sys
import os

# Start from the current working directory (where notebook is)
cwd = os.getcwd()

# Go two levels up (to the "grandparent")
grandparent_dir = os.path.abspath(os.path.join(cwd, "..", ".."))

# Add to sys.path if not already there
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from SpotmicroEnv import SpotmicroEnv
from reward_function import reward_function, RewardState

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ========= CONFIG ==========
TOTAL_STEPS = 5_000_000
run = "stand"
log_dir = f"./logs/{run}"

def clipped_linear_schedule(initial_value, min_value=1e-5):
    def schedule(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return schedule

checkpoint_callback = CheckpointCallback(
    save_freq=TOTAL_STEPS // 40,
    save_path=f"{run}_checkpoints",
    name_prefix=f"ppo_{run}"
)

# ========= ENV ==========
train_env = DummyVecEnv([lambda: SpotmicroEnv(use_gui=False, reward_fn=reward_function, reward_state=RewardState(), dest_save_file=f"states/{run}.pkl")])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.)

# ========= MODEL ==========
model = PPO(
    "MlpPolicy", 
    train_env,
    verbose=0,   # no default printouts
    learning_rate=clipped_linear_schedule(3e-4),
    ent_coef=0.001,
    clip_range=0.1,
    tensorboard_log=log_dir,
)

# Custom logger: ONLY csv + tensorboard (no stdout table)
new_logger = configure(log_dir, ["csv", "tensorboard"])
model.set_logger(new_logger)

# ========= TRAIN ==========
model.learn(
    total_timesteps=TOTAL_STEPS,
    reset_num_timesteps=False,
    callback=checkpoint_callback
)
model.save(f"ppo_{run}")
train_env.save(f"{run}_vecnormalize.pkl")
train_env.close()