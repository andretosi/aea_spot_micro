"""
This script only serves the purpose of showing what a robot does when acting randomly.
It's actually quite useful to compare agains results obtained with a trained policy
"""

import time

from spotmicro.env.spotmicro_env import SpotmicroEnv
from reward_functions.walking_reward_function import reward_function, RewardState
from spotmicro.agent.controller import Controller

ctrl = Controller.from_randomController()
env = SpotmicroEnv(ctrl, use_gui=True, reward_fn=reward_function, reward_state=RewardState())
obs, _ = env.reset()
print("RESET")
time.sleep(2)
print("GO")
for _ in range(3001):
    action = env.action_space.sample()  # Take a random action
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1.0 / 60.0)  # Slow down simulation for visualization
    if terminated or truncated:
        obs, _ = env.reset()