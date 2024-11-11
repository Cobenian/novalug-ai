import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from termcolor import cprint
from novalug.baseball.baseball_rules import BaseballRules


class PitchingEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, game_state, baseball_rules=BaseballRules()):
        super().__init__()
        N_DISCRETE_ACTIONS = len(game_state.get_pitcher_skills()) * len(
            game_state.get_pitch_intents()
        )
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # max values for the observation space
        pitch_skills = [10] * len(game_state.get_pitcher_skills())
        batters_skills = [10] * len(game_state.get_batters())
        batters_count = len(game_state.get_batters())
        discrete_array = (
            pitch_skills
            + batters_skills
            + [
                batters_count,
                baseball_rules.number_of_innings,
                baseball_rules.max_runs,
                baseball_rules.outs_per_inning,
                baseball_rules.max_pitch_count,
                baseball_rules.balls_per_plate_appearance,
                baseball_rules.strikes_per_plate_appearance,
                baseball_rules.runner_on_first,
                baseball_rules.runner_on_second,
                baseball_rules.runner_on_third,
            ]
        )
        self.observation_space = spaces.MultiDiscrete(discrete_array)

        self._game_state = game_state
        self._baseball_rules = baseball_rules

    def step(self, action):
        print("\ttried to throw pitch:", self._game_state.action_description(action))
        pitch_outcome = self.calculate_pitch_outcome(action)
        play_outcome = self._game_state.update_game_state_for_play(pitch_outcome)
        reward = self.get_reward_for_play_outcome(play_outcome)
        observation = self.get_obs()
        info = self.get_info()
        terminated = self._game_state.game_is_over(self._baseball_rules)
        truncated = self._game_state.game_in_invalid_state(self._baseball_rules)
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        cprint("START OF GAME!!!!", "blue")
        self._game_state.set_initial_values()
        observation = self.get_obs()
        info = self.get_info()
        return observation, info

    def render(self):
        self._game_state.print_count()

    def close(self):
        pass

    # our stuff

    def get_obs(self):
        obs = (
            self._game_state.get_pitcher_skills()
            + self._game_state.get_batters()
            + [
                self._game_state.get_current_batter(),
                self._game_state.get_current_inning(),
                self._game_state.get_current_runs(),
                self._game_state.get_current_outs(),
                self._game_state.get_current_pitch_count(),
                self._game_state.get_current_balls(),
                self._game_state.get_current_strikes(),
                self._game_state.get_current_runner_on_first(),
                self._game_state.get_current_runner_on_second(),
                self._game_state.get_current_runner_on_third(),
            ]
        )
        return np.array(obs)

    def get_info(self):
        return {}

    # OUR IMPL DETAILS

    def get_reward_for_play_outcome(self, outcome):
        if outcome == "non_pitch":
            return -100
        elif outcome == "hit":
            return -5
        elif outcome == "walk":
            return -10
        elif outcome == "ball":
            return 0
        elif outcome == "strike":
            return 50
        elif outcome == "out":
            return 100

    def get_game_state(self):
        return self._game_state

    def calculate_pitch_outcome(self, action):
        # THIS WILL RETURN ONE OF:
        # -1 is a pitch this pitcher doesn't throw
        # 0 is a ball in play - hit
        # 1 is a ball
        # 2 is a strike
        # 3 is a ball in play - out
        pitcher_tried_to_throw_a_ball = (
            action % len(self._game_state.get_pitch_intents()) == 0
        )
        if action <= 0:
            pitcher_skill_index = 0
        else:
            pitcher_skill_index = int(action) // len(
                self._game_state.get_pitch_intents()
            )
        pitcher_skill_at_pitch = self._game_state.get_pitcher_skills()[
            pitcher_skill_index
        ]
        batter_skill = self._game_state.get_batters()[
            self._game_state.get_current_batter()
        ]
        defense_skill = self._game_state.get_defense_skill()

        if pitcher_skill_at_pitch == 0:
            return -1

        if pitcher_tried_to_throw_a_ball:
            hit_chances = [0] * batter_skill
            # since a ball was thrown, the batter has a lower chance of hitting
            half_length = len(hit_chances) // 4
            hit_chances = hit_chances[:half_length]
            ball_chances = [1] * pitcher_skill_at_pitch
            strike_chances = [2] * (10 - pitcher_skill_at_pitch)
        else:
            hit_chances = [0] * batter_skill
            ball_chances = [1] * (10 - pitcher_skill_at_pitch)
            strike_chances = [2] * pitcher_skill_at_pitch

        chances = hit_chances + ball_chances + strike_chances
        result = random.choice(chances)
        if result == 0:
            # ball was hit, but did the defense get an out?
            hit_chances = [0] * (10 - defense_skill)
            out_chances = [3] * defense_skill
            return random.choice(hit_chances + out_chances)
        else:
            return result


class PitcherTrainingEnv(PitchingEnv):
    def reset(self, seed=None, options=None):
        cprint("START OF TRAINING!!!!", "blue")
        self._game_state.randomize_offense_and_defense()
        self._game_state.set_initial_values()
        observation = self.get_obs()
        info = self.get_info()
        return observation, info
