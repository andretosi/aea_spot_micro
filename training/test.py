"""
This script only serves the purpose of showing what a robot does when acting randomly.
It's actually quite useful to compare agains results obtained with a trained policy
"""

import time

from src.spotmicro.env.spotmicro_env_mujoco import SpotmicroEnv
from reward_functions.walking_reward_function import reward_function, RewardState
from src.spotmicro import RandomController
from src.spotmicro.tools.config import Config

cfg = Config("configs/test_config.yaml")
dev = RandomController(cfg)
env = SpotmicroEnv(dev, cfg, reward_function, RewardState(), use_gui=True, )
obs, _ = env.reset()

cfg.save("configs/test_config2.yaml")

for _ in range(3001):
    action = env.action_space.sample()  # Take a random action
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1.0 / 60.0)  # Slow down simulation for visualization
    if terminated or truncated:
        obs, _ = env.reset()