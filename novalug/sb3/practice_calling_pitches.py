from stable_baselines3 import PPO

from novalug.sb3.pitching_env import PitchingEnv
from novalug.baseball.game_state import GameState


def choose_a_pitcher():
    pitcher_skills = [
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
    return pitcher_skills, pitch_intents


def choose_batters():
    batters = [
        4,  # batter 1 skill
        3,  # batter 2 skill
        4,  # batter 3 skill
        3,  # batter 4 skill
        3,  # batter 5 skill
        2,  # batter 6 skill
        3,  # batter 7 skill
        1,  # batter 8 skill
        2,  # batter 9 skill
    ]
    return batters


def choose_defense():
    defense_skill = 7
    return defense_skill


def make_game_environment(game_state):
    env = PitchingEnv(game_state)
    return env


def simulate_game(model_file, env):
    model = PPO.load(model_file, env=env)

    obs = model.get_env().reset()
    while True:
        env.render()
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated:
            env.get_game_state().print_full()
            break


def main():
    model_file = "models/sb3/pitching_model_trained.zip"
    pitcher_skill, pitch_intents = choose_a_pitcher()
    batters = choose_batters()
    defense_skill = choose_defense()
    game_state = GameState(pitcher_skill, pitch_intents, batters, defense_skill)
    env = make_game_environment(game_state)
    simulate_game(model_file, env)
