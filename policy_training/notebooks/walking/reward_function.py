import pybullet
import numpy as np
from SpotmicroEnv import SpotmicroEnv

class RewardState:
    def __init__(self):
        self.prev_contacts = set()
        self.prev_base_position = np.array([0.0, 0.0, 0.0])

    def populate(self, env: SpotmicroEnv):
        return

def fade_in(current_step, start, scale=2.0):
    if current_step < start:
        return 0.0
    return 1.0 - np.exp(-scale * (current_step - start) / 1_000_000)

def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:

    roll, pitch, _ = env.agent.state.roll_pitch_yaw
    percentage_error = 0.2

    # Errors and metrics
    lin_vel_sq_perc_error = (np.linalg.norm(env.target_lin_velocity - env.agent.state.linear_velocity) / np.linalg.norm(env.target_lin_velocity) + 1e-6)** 2
    ang_vel_error = np.linalg.norm((env.target_ang_velocity - env.agent.state.angular_velocity))** 2
    deviation_penalty = np.linalg.norm(env.agent.state.joint_positions - env.agent.homing_positions) ** 2
    height_penalty = (env.agent.state.base_position[2] - env.config.target_height) ** 2
    action_rate = np.mean(action - env.agent.previous_action) ** 2
    vertical_velocity_sq =  env.agent.state.linear_velocity[2] ** 2
    stabilization_penalty = roll ** 2 + pitch ** 2
    perp_velocity = env.agent.state.linear_velocity - ((np.dot(env.agent.state.linear_velocity, env.target_lin_velocity) / (np.linalg.norm(env.target_lin_velocity) ** 2)) * env.target_lin_velocity)
    total_normalized_effort = np.sum([(j.effort / j.max_torque) ** 2 for j in env.agent.motor_joints]) / len(env.agent.motor_joints)

    # Derived penalties
    tolerance = 0.3
    lin_vel_reward = max(1.0 - ((lin_vel_sq_perc_error / tolerance) **2), -1.0)
    drift_penalty = np.linalg.norm(perp_velocity) ** 2

    delta_pos = env.agent.state.base_position - env.reward_state.prev_base_position
    progress = np.dot(delta_pos, env.target_lin_velocity) / (np.linalg.norm(env.target_lin_velocity)+1e-6)
    # clip to a small range each step
    progress = np.clip(progress / env.sim_frequency, -0.5, 0.5)

    # === Final Reward ===
    reward_dict = {
        "linear_vel_reward": 10 * lin_vel_reward,
        "progress_reward": 1.5 * progress,
        "angular_vel_penalty": -5 * ang_vel_error,
        "drift_penalty": -6 * drift_penalty,
        "action_rate_penalty": -3 * action_rate,
        "height_penalty": -3 * min(height_penalty, 1.0),
        "stabilization_penalty": -3 * min(stabilization_penalty, 1.0),
        "effort_penalty": -1.5 * total_normalized_effort,
        "deviation_penalty": -0.0 * deviation_penalty,
        "vertical_motion_penalty": -0.5 * vertical_velocity_sq,
    }
    total_reward = sum(reward_dict.values())

    env.log_rewards(reward_dict)
    return total_reward, reward_dict
