import time
import numpy as np
from stable_baselines3 import PPO
from SpotmicroEnv import SpotmicroEnv
from walking_reward_function import reward_function, RewardState

run = "walk27M-0"

env = SpotmicroEnv(
    use_gui=True, 
    reward_fn=reward_function,
    reward_state=RewardState(),
    src_save_file=f"states/{run}.pkl"
    )
obs, _ = env.reset()

# Load your trained model
model = PPO.load(f"policies/ppo_{run}")  # or path to your .zip
#model = PPO.load(f"policies/{run}_checkpoints/ppo_{run}_22002496_steps.zip")
print(f"num steps: {env.num_steps}")

# Run rollout
for _ in range(3001):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Terminated")
        env.plot_reward_components()  # 👈 plot per episode
        obs, _ = env.reset()
    
    time.sleep(1/60.)  # Match simulation step time for real-time playback

env.close()