from stable_baselines3.common.env_checker import check_env

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from novalug.sb3.pitching_env import PitchingEnv

pitcher_skill = [
    8,  # fastball
    3,  # curveball
    7,  # slider
    0,  # changeup
    0,  # knuckleball
    0,  # splitter
    0,  # sinker
    0,  # cutter
]

# batters = [8, 9, 7, 7, 4, 5, 3, 4, 4]
batters = [4, 3, 2, 2, 1, 5, 2, 1, 3]

defense_skill = 7

# Instantiate the env
env = PitchingEnv(pitcher_skill, batters, defense_skill)

# check_env(env)

# model_file = "models/pitching_model_undertrained.zip"
model_file = "models/pitching_model_trained.zip"
# model_file = "models/pitching_model_trained_2.zip"

model = PPO.load(model_file, env=env)

# env.render()

vec_env = model.get_env()
obs = vec_env.reset()
while True:
    env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated:
        env.print_full()
        break
