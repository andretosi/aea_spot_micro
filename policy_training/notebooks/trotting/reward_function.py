import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

class RewardState:
    def __init__(self):
        self.prev_contacts = set()

    def populate(self, env: SpotmicroEnv):
        return

def fade_in(current_step, start, scale=2.0):
    if current_step < start:
        return 0.0
    return 1.0 - np.exp(-scale * (current_step - start) / 1_000_000)

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:

    roll, pitch, _ = env.agent.state.roll_pitch_yaw

    # Errors
    lin_vel_error = np.linalg.norm(env.target_lin_velocity - env.agent.state.linear_velocity) ** 2
    ang_vel_error = np.linalg.norm((env.target_ang_velocity - env.agent.state.angular_velocity) / env.config.max_angular_velocity)** 2
    deviation_penalty = np.linalg.norm(env.agent.state.joint_positions - env.agent.homing_positions) ** 2
    height_penalty = (env.agent.state.base_position[2] - env.config.target_height) ** 2
    action_rate = np.mean(action - env.agent.previous_action) ** 2
    vertical_velocity_sq =  env.agent.state.linear_velocity[2] ** 2
    stabilization_penalty = roll ** 2 + pitch ** 2
    perp_velocity = env.agent.state.linear_velocity - ((np.dot(env.agent.state.linear_velocity, env.target_lin_velocity) / (np.linalg.norm(env.target_lin_velocity) ** 2)) * env.target_lin_velocity)
    total_normalized_effort = np.sum([(j.effort / j.max_torque) ** 2 for j in env.agent.motor_joints]) / len(env.agent.motor_joints)

    # Derived penalties
    lin_vel_reward = max(1 - 1.75 * lin_vel_error, -1.0)
    drift_penalty = np.linalg.norm(perp_velocity) ** 2


    # === Final Reward ===
    reward_dict = {
        "linear_vel_reward": 10 * lin_vel_reward,
        "height_penalty": -3 * min(height_penalty, 1.0),
        "stabilization_penalty": -3 * min(stabilization_penalty, 1.0),
        "drift_penalty": -3 * fade_in(env.num_steps, 7_000_000, scale=2.5) * drift_penalty,
        "angular_vel_penalty": -1.5 * ang_vel_error,
        "effort_penalty": -1.5 * fade_in(env.num_steps, 7_000_000, scale=2.5) * total_normalized_effort,
        "deviation_penalty": -0.5 * deviation_penalty,
    }
    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict
