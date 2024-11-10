from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from novalug.sb3.pitching_env import PitcherTrainingEnv
from novalug.li.game_state import GameState

# these values do not matter for training. Difference runs will use randomized values

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

pitch_intents = [
    0,  # strike
    0,  # strike
    1,  # ball
]

batters = [
    7,  # batter 1 skill
    7,  # batter 2 skill
    7,  # batter 3 skill
    7,  # batter 4 skill
    7,  # batter 5 skill
    7,  # batter 6 skill
    7,  # batter 7 skill
    7,  # batter 8 skill
    7,  # batter 9 skill
]

defense_skill = 6

# Instantiate the env
game_state = GameState(pitcher_skill, pitch_intents, batters, defense_skill)
env = PitcherTrainingEnv(game_state)

check_env(env)

steps = 10_000_000

# model_file = "models/pitching_model_undertrained.zip"
model_file = "models/pitching_model_trained.zip"

# Define and Train the agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=steps)
model.save(model_file)
