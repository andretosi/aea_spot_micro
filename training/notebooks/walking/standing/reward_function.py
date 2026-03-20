import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

class RewardState:
    def __init__(self):
        self.prev_contacts = set()
    
    def populate(self, env: SpotmicroEnv):
        return

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:

    # === Action Sparsity ===
    action_reward = 1.0 - float(np.mean(np.abs(action)))


    # === Final Reward ===
    reward_dict = {
        "action_reward": 2 * action_reward,
    }
    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict
