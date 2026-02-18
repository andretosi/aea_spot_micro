"""
This script only serves the purpose of showing what a robot does when acting randomly.
It's actually quite useful to compare agains results obtained with a trained policy
"""

import time

from spotmicro.env.spotmicro_env import SpotmicroEnv
from reward_functions.walking_reward_function import reward_function, RewardState
from spotmicro.devices.random_controller import RandomController
from spotmicro.tools.config import Config

cfg = Config()
dev = RandomController(cfg)
env = SpotmicroEnv(dev, cfg, reward_function, use_gui=False, reward_state=RewardState())
obs, _ = env.reset()


for _ in range(3001):
    time.sleep(1)
    action = env.action_space.sample()  # Take a random action
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1.0 / 60.0)  # Slow down simulation for visualization
    if terminated or truncated:
        obs, _ = env.reset()