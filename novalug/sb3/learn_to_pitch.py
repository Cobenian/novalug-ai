from stable_baselines3.common.env_checker import check_env

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from novalug.sb3.pitching_env import PitchingEnv

pitcher_1_skill = [
    8,  # fastball
    3,  # curveball
    7,  # slider
    3,  # changeup
    7,  # knuckleball
    7,  # splitter
    5,  # sinker
    4,  # cutter
]

batters = [4, 3, 7, 7, 4, 5, 3, 4, 2]

defense_skill = 9

# print('let us start')

# Instantiate the env
env = PitchingEnv(pitcher_1_skill, batters, defense_skill)

# print('created env')

check_env(env)

# print('checked env')

# Define and Train the agent
# model = A2C("CnnPolicy", env).learn(total_timesteps=1000)

# print('time to create model')
model = PPO("MlpPolicy", env, verbose=1)
# print('created model')
model.learn(total_timesteps=10000)
# print('trained model')

env.render()

# print('start to run')
vec_env = model.get_env()
obs = vec_env.reset()
# print('reset env obs', obs)
while True:
    action, _states = model.predict(obs)
    # print('pitch thrown', env.action_description(action))
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
    print("\n")
    if terminated:
        break
