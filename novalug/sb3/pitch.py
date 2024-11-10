from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from novalug.sb3.pitching_env import PitchingEnv
from novalug.li.game_state import GameState

# model_file = "models/pitching_model_undertrained.zip"
model_file = "models/pitching_model_trained.zip"

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

pitch_intents = [
    0,  # strike
    0,  # strike
    1,  # ball
]

batters = [8, 9, 7, 7, 4, 5, 3, 4, 4]

defense_skill = 7

game_state = GameState(pitcher_skill, pitch_intents, batters, defense_skill)

# Instantiate the env
env = PitchingEnv(game_state)
check_env(env)

model = PPO.load(model_file, env=env)

obs = model.get_env().reset()
while True:
    env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated:
        env.get_game_state().print_full()
        break
