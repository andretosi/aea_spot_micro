"""
This script only serves the purpose of showing what a robot does when acting randomly.
It's actually quite useful to compare agains results obtained with a trained policy
"""

from SpotmicroEnv import SpotmicroEnv
import time
from walking_reward_function import reward_function, RewardState

env = SpotmicroEnv(use_gui=True, reward_fn=reward_function, reward_state=RewardState())
obs, _ = env.reset()

for _ in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    time.sleep(5.)

    if terminated or truncated:
        obs, _ = env.reset()