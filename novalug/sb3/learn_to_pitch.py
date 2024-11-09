from stable_baselines3.common.env_checker import check_env

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from novalug.sb3.pitching_env import PitcherTrainingEnv

pitcher_skill = [
    8,  # fastball
    8,  # curveball
    8,  # slider
    8,  # changeup
    8,  # knuckleball
    8,  # splitter
    8,  # sinker
    8,  # cutter
]

batters = [7, 7, 7, 7, 7, 7, 7, 7, 7]

defense_skill = 6

# Instantiate the env
env = PitcherTrainingEnv(pitcher_skill, batters, defense_skill)

check_env(env)

# steps = 1000
# steps = 15_000_000
steps = 15_000_000

# model_file = "models/pitching_model_undertrained.zip"
# model_file = "models/pitching_model_trained.zip"
model_file = "models/pitching_model_trained_well.zip"

# Define and Train the agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=steps)

model.save(model_file)

# env.render()

# vec_env = model.get_env()
# obs = vec_env.reset()
# while True:
#     env.render()
#     action, _states = model.predict(obs)
#     obs, rewards, terminated, truncated, info = env.step(action)
#     if terminated:
#         env.print_full()
#         break
