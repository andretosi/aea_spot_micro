from stable_baselines3 import PPO
from SpotmicroEnv import SpotmicroEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from walking_reward_function import reward_function, init_custom_state
from torch.utils.tensorboard import SummaryWriter

TOTAL_STEPS = 8_000_000
run = "stand1M-2"

def clipped_linear_schedule(initial_value, min_value=1e-5):
    def schedule(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return schedule

checkpoint_callback = CheckpointCallback(
    save_freq=TOTAL_STEPS / 20,                
    save_path=f"./policies/{run}_checkpoints",  # Folder to save in
    name_prefix=f"ppo_{run}"            # File name prefix
)

#writer = SummaryWriter(log_dir=f"./logs/reward_components/{run}")
env = SpotmicroEnv(use_gui=False, reward_fn=reward_function, dest_save_file=f"states/state{run}.pkl", writer=None)
check_env(env, warn=True) #optional

model = PPO(
    "MlpPolicy", 
    env, 
    verbose = 1, 
    n_steps=2048,
    batch_size=512,
    learning_rate=clipped_linear_schedule(3e-4),
    ent_coef=0.0035, #previously 0.0015
    clip_range=0.15,
    tensorboard_log="./logs"
)
model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint_callback)
model.save(f"policies/ppo_{run}")
env.close()
