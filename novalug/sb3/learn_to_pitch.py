from stable_baselines3.common.env_checker import check_env

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from novalug.sb3.pitching_env import PitchingEnv

pitcher_skill = [
    8,  # fastball
    3,  # curveball
    7,  # slider
    # 3,  # changeup
    # 7,  # knuckleball
    # 7,  # splitter
    # 5,  # sinker
    # 4,  # cutter
]

batters = [4, 3, 7, 7, 4, 5, 3, 4, 2]

defense_skill = 5

# Instantiate the env
env = PitchingEnv(pitcher_skill, batters, defense_skill)

check_env(env)

# Define and Train the agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=250000)

env.render()

vec_env = model.get_env()
obs = vec_env.reset()
while True:
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated:
        env.print_full()
        break
