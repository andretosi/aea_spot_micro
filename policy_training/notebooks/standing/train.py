import time
import numpy as np
from stable_baselines3 import PPO
import sys
import os

# Start from the current working directory (where notebook is)
cwd = os.getcwd()
grandparent_dir = os.path.abspath(os.path.join(cwd, "..", ".."))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from SpotmicroEnv import SpotmicroEnv
from reward_function import reward_function, RewardState

from stable_baselines3.common.callbacks import CheckpointCallback
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

# ========= CUSTOM CALLBACK ==========
class VecNormalizeCheckpoint(CheckpointCallback):
    """
    Saves model + VecNormalize stats at checkpoint intervals
    """
    def _on_step(self) -> bool:
        # Save model as usual
        super()._on_step()
        # Save VecNormalize stats if applicable
        env = self.model.get_env()
        if hasattr(env, "save") and isinstance(env, VecNormalize):
            vec_path = os.path.join(self.save_path, f"vecnormalize_step_{self.num_timesteps}.pkl")
            os.makedirs(self.save_path, exist_ok=True)
            env.save(vec_path)
        return True

checkpoint_callback = VecNormalizeCheckpoint(
    save_freq=TOTAL_STEPS // 40,
    save_path=f"{run}_checkpoints",
    name_prefix=f"ppo_{run}"
)

# ========= ENV ==========
train_env = DummyVecEnv([
    lambda: SpotmicroEnv(
        use_gui=False,
        reward_fn=reward_function,
        reward_state=RewardState(),
        dest_save_file=f"states/{run}.pkl"
    )
])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.)

# ========= MODEL ==========
model = PPO(
    "MlpPolicy", 
    train_env,
    verbose=0,
    learning_rate=clipped_linear_schedule(3e-4),
    ent_coef=0.001,
    clip_range=0.1,
    tensorboard_log=log_dir,
)

# Custom logger: ONLY csv + tensorboard
new_logger = configure(log_dir, ["csv", "tensorboard"])
model.set_logger(new_logger)

# ========= TRAIN ==========
model.learn(
    total_timesteps=TOTAL_STEPS,
    reset_num_timesteps=False,
    callback=checkpoint_callback
)

# Final save (model + VecNormalize)
model.save(f"ppo_{run}")
train_env.save(f"{run}_vecnormalize.pkl")
train_env.close()
