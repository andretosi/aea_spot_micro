import time
import numpy as np
from stable_baselines3 import PPO
from SpotmicroEnv import SpotmicroEnv
from standing_reward_function import reward_function, init_custom_state

env = SpotmicroEnv(
    use_gui=True, 
    reward_fn=reward_function,
    init_custom_state=init_custom_state,
    src_save_file="states/state2M-2.pkl"
    )
obs, _ = env.reset()

# Load your trained model
model = PPO.load(f"policies/ppo_stand2M-2")  # or path to your .zip
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