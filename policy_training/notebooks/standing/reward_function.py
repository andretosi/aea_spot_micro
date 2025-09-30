import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

class RewardState:
    def __init__(self):
        self.prev_contacts = set()
        self.a = -2
        self.b = 0
        self.c = 0
    
    def populate(self, env: SpotmicroEnv):
        norm_hp = np.array([float(j.from_position_to_action(j.homing_position)) for j in env.agent.motor_joints])
        self.b = -2 * self.a * norm_hp
        self.c = 1 - 2 * np.square(norm_hp) + 4*norm_hp
        return

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:

    rewards = np.clip(np.array([env.reward_state.a * a**2 + env.reward_state.b * a + env.reward_state.c for a in action]), -0.5, 1.0)
    action_reward = np.mean(rewards)
    #MIGHT TRY WITH JOINT POSITIONS INSTEAD OF ACTIONS

    # === Final Reward ===
    reward_dict = {
        "action_reward": 2 * action_reward,
    }
    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict
