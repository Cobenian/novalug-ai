from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from novalug.sb3.pitching_env import PitcherTrainingEnv
from novalug.baseball.game_state import GameState


def choose_a_pitcher():
    pitcher_skills = [
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
    return pitcher_skills, pitch_intents


def choose_batters():
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
    return batters


def choose_defense():
    defense_skill = 6
    return defense_skill


def make_game_environment(pitcher_skill, pitch_intents, batters, defense_skill):
    # Instantiate the env with the game state (which keeps tracks of the score, inning, outs, balls, strikes, etc)
    game_state = GameState(pitcher_skill, pitch_intents, batters, defense_skill)
    env = PitcherTrainingEnv(game_state)
    # this is a sanity check to make sure the environment is set up correctly
    check_env(env)
    return env


def learn_to_pitch(env, steps, model_filename):
    # Define and Train the agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=steps)
    model.save(model_filename)


def main():
    # NOTE: the values for the pitchers, batters and defense do not matter for training. Each batch of training runs will use randomized values
    pitcher_skills, pitch_intents = choose_a_pitcher()
    batters = choose_batters()
    defense_skill = choose_defense()

    env = make_game_environment(pitcher_skills, pitch_intents, batters, defense_skill)
    training_steps = 25_000_000
    model_file = "models/sb3/pitching_model_trained.zip"
    learn_to_pitch(env, training_steps, model_file)
