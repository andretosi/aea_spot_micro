from stable_baselines3 import PPO
from SpotmicroEnv import SpotmicroEnv
from stable_baselines3.common.env_checker import check_env
from reward_function import reward_function, init_custom_state
from torch.utils.tensorboard import SummaryWriter


TOTAL_STEPS = 20_000_000

def clipped_linear_schedule(initial_value, min_value=1e-5):
    def schedule(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return schedule

run = "walk20M-9"
#writer = SummaryWriter(log_dir=f"./logs/reward_components/{run}")
env = SpotmicroEnv(use_gui=False, reward_fn=reward_function, init_custom_state=init_custom_state, dest_save_file=f"states/state{run}.pkl", writer=None)
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
model.learn(total_timesteps=TOTAL_STEPS)
model.save(f"policies/ppo_{run}")
env.close()
