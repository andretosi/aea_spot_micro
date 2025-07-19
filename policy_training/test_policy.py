import time
import numpy as np
from stable_baselines3 import PPO
from SpotmicroEnv import SpotmicroEnv
from standing_reward_function import reward_function, init_custom_state

run = "stand1M-0_800000_steps"
env = SpotmicroEnv(
    use_gui=True, 
    reward_fn=reward_function#, 
    #init_custom_state=init_custom_state#, 
    #src_save_file=f"states/state{run}.pkl"
    )
obs, _ = env.reset()

# Load your trained model
model = PPO.load(f"policies/stand1M-0_checkpoints/ppo_{run}")  # or path to your .zip
env._total_steps_counter = 800_000
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