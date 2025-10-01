import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

class RewardState:
    def __init__(self):
        self.prev_contacts = set()
        self.perc_error = 0.02
        self.ms = []
        self.q = 1/(1+self.perc_error)
    
    def populate(self, env: SpotmicroEnv):
        norm_hp = np.array([float(j.from_position_to_action(j.homing_position)) for j in env.agent.motor_joints])
        for hp in norm_hp:
            self.ms.append(-1 / ((1-self.perc_error) * hp))

        

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:

    rewards = np.clip(np.array([1-abs(m*a + env.reward_state.q) for m, a in zip(env.reward_state.ms, action)]), -0.5, 1.0)
    action_reward = np.mean(rewards)

    # === Final Reward ===
    reward_dict = {
        "action_reward": 2 * action_reward,
    }
    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict
