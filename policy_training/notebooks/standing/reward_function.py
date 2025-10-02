import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

class RewardState:
    def __init__(self):
        self.prev_contacts = set()
        self.a = -5.0   # controls steepness of parabola

        # will hold homing positions in raw joint space
        self.homing_positions = None
    
    def populate(self, env: SpotmicroEnv):
        # Store raw homing positions for each motor joint
        self.homing_positions = np.array([
            float(j.homing_position) for j in env.agent.motor_joints
        ])
        return

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:
    # === Work in joint space directly ===
    positions = np.array(env.agent.state.joint.positions)

    # Parabola centered at homing position
    diffs = positions - env.reward_state.homing_positions
    rewards = env.reward_state.a * np.square(diffs) + 1.0

    # Clip to desired range
    rewards = np.clip(rewards, -0.5, 1.0)

    pos_reward = np.mean(rewards)

    # === Final Reward ===
    reward_dict = {
        "position_reward": 2.0 * pos_reward,
    }
    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict
