import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

class RewardState:
    def __init__(self):
        self.prev_contacts = set()
        self.perc_error = 0.02
        self.homing_positions = None  # store normalized homing positions
        self.tol = None               # tolerance per joint

    def populate(self, env: SpotmicroEnv):
        norm_hp = np.array([
            float(j.from_position_to_action(j.homing_position))
            for j in env.agent.motor_joints
        ])
        self.homing_positions = norm_hp
        self.tol = self.perc_error * norm_hp  # ±2% tolerance per joint


def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:
    hp = env.reward_state.homing_positions
    tol = env.reward_state.tol

    # Triangular reward: peak=1 at hp, falls linearly to 0 at hp ± tol
    rewards = 1 - np.abs((action - hp) / tol)
    rewards = np.clip(rewards, -0.25, 1.0)

    action_reward = float(np.mean(rewards))

    # === Final Reward ===
    reward_dict = {
        "action_reward": 2 * action_reward,  # scaling factor
    }
    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict
