from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from pathlib import Path

from spotmicro.env.spotmicro_env import SpotmicroEnv
from reward_functions.walking_reward_function import reward_function, RewardState

TOTAL_STEPS = 3_000_000
run = "stand"
DATA_DIR =  Path("data") / f"{run}_results"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # ensure directory exists

def clipped_linear_schedule(initial_value, min_value=1e-5):
    def schedule(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return schedule

checkpoint_callback = CheckpointCallback(
    save_freq=TOTAL_STEPS / 10,                
    save_path=str(DATA_DIR / "checkpoints"),  # Folder to save in
    name_prefix=f"ppo_{run}"            # File name prefix
)

env = SpotmicroEnv(
    use_gui=False,
    reward_fn=reward_function, 
    reward_state=RewardState(), 
    dest_save_file=str(DATA_DIR / f"{run}.pkl")
    )
check_env(env, warn=True) #optional

model = PPO(
    "MlpPolicy", 
    env, 
    verbose = 1, 
    learning_rate=clipped_linear_schedule(3e-4),
    ent_coef=0.002, #previously 0.0015
    clip_range=0.1,
    tensorboard_log=str(DATA_DIR / "logs"),
    )

model.learn(
    total_timesteps=TOTAL_STEPS,
    reset_num_timesteps=False,
    callback=checkpoint_callback
    )
model.save(str(DATA_DIR / f"ppo_{run}_final"))
env.close()
