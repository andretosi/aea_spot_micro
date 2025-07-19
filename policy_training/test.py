"""
This script only serves the purpose of showing what a robot does when acting randomly.
It's actually quite useful to compare agains results obtained with a trained policy
"""

from SpotmicroEnv import SpotmicroEnv
import time
from reward_function import reward_function, init_custom_state

env = SpotmicroEnv(use_gui=True, reward_fn=reward_function, init_custom_state=init_custom_state)
obs, _ = env.reset()

for _ in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"distance from target height: {env.agent_base_position[2] - env.TARGET_HEIGHT}")
    time.sleep(1/60.)

    if terminated or truncated:
        obs, _ = env.reset()