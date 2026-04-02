import time
from stable_baselines3 import PPO

from spotmicro.env.spotmicro_env import SpotmicroEnv
from spotmicro.physics.factory import create_backend
from spotmicro.devices.fixed_controller import FixedController
from spotmicro.tools.config import Config
from reward_functions.standing_reward_function import reward_function, RewardState
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

# ========= ENV ==========
cfg = Config()
dev = FixedController("still") #not a configurable class
backend = create_backend("pybullet", use_gui=False)
env = SpotmicroEnv(
    backend,
    dev,
    cfg,
    reward_function,
    RewardState(),
    use_gui=False
)

for _ in range(3001):
    action = env.action_space.sample()  # Take a random action
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1.0 / 60.0)  # Slow down simulation for visualization
    if terminated or truncated:
        obs, _ = env.reset()