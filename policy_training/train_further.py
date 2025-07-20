from SpotmicroEnv import SpotmicroEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from standing_reward_function import reward_function, init_custom_state
from stable_baselines3.common.callbacks import CheckpointCallback

TOTAL_STEPS = 1_000_000
run = "stand2M-2"


def clipped_linear_schedule(initial_value, min_value=1e-5):
    def schedule(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return schedule

checkpoint_callback = CheckpointCallback(
    save_freq=TOTAL_STEPS / 20,                
    save_path=f"./policies/{run}_checkpoints",  # Folder to save in
    name_prefix=f"ppo_{run}"            # File name prefix
)

env = SpotmicroEnv(
    use_gui=False,
    reward_fn=reward_function, 
    init_custom_state=init_custom_state, 
    src_save_file="states/statestand1M-0.pkl",
    dest_save_file="states/state2M-2.pkl"
    )
check_env(env, warn=True) #optional

model = PPO.load("policies/ppo_stand1M-2")
model.set_env(env)
model.tensorboard_log = "./logs"
model.learn(
    total_timesteps=TOTAL_STEPS,
    reset_num_timesteps=False,
    )
model.save("policies/ppo_stand2M-2")
env.close()