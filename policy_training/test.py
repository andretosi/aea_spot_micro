"""
This script only serves the purpose of showing what a robot does when acting randomly.
It's actually quite useful to compare agains results obtained with a trained policy
"""

from SpotmicroEnv import SpotmicroEnv
import time
from walking_reward_function import reward_function, RewardState

env = SpotmicroEnv(use_gui=True, reward_fn=reward_function, reward_state=RewardState())
obs, _ = env.reset()

while True:
    print(env.agent.state.base_position[2]) 