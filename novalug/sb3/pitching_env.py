import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random


class PitchingEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, pitcher_skill, batters, defense_skill, innings=9):
        super().__init__()
        pitch_intents = [
            0,  # "Strike",
            0,  # "Strike",
            1,  # "Ball",
        ]
        N_DISCRETE_ACTIONS = len(pitcher_skill) * len(pitch_intents)
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # max values for the observation space
        max_pitch_count = 999
        max_runs = 999
        batters_count = len(batters)
        # defined by the rules of baseball
        outs = 3
        balls = 4
        strikes = 3
        runner_on_first = 2  # boolean
        runner_on_second = 2  # boolean
        runner_on_third = 2  # boolean
        pitch_skills = [10] * len(pitcher_skill)
        batters_skills = [10] * len(batters)
        discrete_array = (
            [innings, max_runs, outs, max_pitch_count]
            + pitch_skills
            + [batters_count]
            + batters_skills
            + [
                balls,
                strikes,
                runner_on_first,
                runner_on_second,
                runner_on_third,
            ]
        )
        self.observation_space = spaces.MultiDiscrete(discrete_array)

        self._innings = innings
        self._batters = batters
        self._pitcher_skill = pitcher_skill
        self._defense_skill = defense_skill
        self._pitch_intents = pitch_intents
        self._max_runs = max_runs
        self.set_initial_values()

    def step(self, action):
        print("pitch thrown", self.action_description(action))
        reward = self.calculate_reward(action)
        self.update_state(reward)
        observation = self.get_obs()
        info = self.get_info()
        terminated = (
            self._current_runs >= self._max_runs
            or self._current_inning >= self._innings
            or (self._current_inning >= self._innings and self._current_outs >= 3)
        )
        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.set_initial_values()
        observation = self.get_obs()
        info = self.get_info()
        return observation, info

    def render(self):
        self.print_count()

    def close(self):
        pass

    # our stuff

    def get_obs(self):
        obs = (
            [
                self._current_inning,
                self._current_runs,
                self._current_outs,
                self._current_pitch_count,
            ]
            + self._pitcher_skill
            + [self._current_batter]
            + self._batters
            + [
                self._current_balls,
                self._current_strikes,
                self._current_runner_on_first,
                self._current_runner_on_second,
                self._current_runner_on_third,
            ]
        )
        return np.array(obs)

    def get_info(self):
        return {}

    def set_initial_values(self):
        self._current_pitch_count = 0
        self._current_total_strikes = 0
        self._current_total_balls = 0
        self._current_inning = 0
        self._current_runs = 0
        self._current_outs = 0
        self._current_balls = 0
        self._current_strikes = 0
        self._current_runner_on_first = 0
        self._current_runner_on_second = 0
        self._current_runner_on_third = 0
        self._current_batter = 0

    def calculate_reward(self, action):
        pitcher_tried_to_throw_a_ball = action % len(self._pitch_intents) == 0
        if action <= 0:
            pitcher_skill_index = 0
        else:
            pitcher_skill_index = int(action) // len(self._pitch_intents)
        pitcher_skill_at_pitch = self._pitcher_skill[pitcher_skill_index]
        batter_skill = self._batters[self._current_batter]
        defense_skill = self._defense_skill

        # -1 is a hit
        # 0 is a ball
        # 1 is a strike
        # 2 is an out on a ball in play

        if pitcher_tried_to_throw_a_ball:
            hit_chances = [-1] * batter_skill
            ball_chances = [0] * (10 - pitcher_skill_at_pitch)
            strike_chances = [1] * pitcher_skill_at_pitch
            out_chances = [2] * defense_skill
        else:
            hit_chances = [-1] * batter_skill
            # since a ball was thrown, the batter has a lower chance of hitting
            half_length = len(hit_chances) // 4
            hit_chances = hit_chances[:half_length]
            ball_chances = [0] * pitcher_skill_at_pitch
            strike_chances = [1] * (10 - pitcher_skill_at_pitch)
            out_chances = [2] * defense_skill

        chances = hit_chances + ball_chances + strike_chances + out_chances
        reward = random.choice(chances)
        return reward

    def update_state(self, reward):
        self._current_pitch_count += 1
        if reward < 0:
            self._current_total_strikes += 1
            print("reward was a hit")
            # hit
            self.advance_runners()
            self.next_batter()
        elif reward == 0:
            print("reward was a ball")
            # ball
            self._current_total_balls += 1
            self._current_balls += 1
            if self._current_balls >= 4:
                self.next_batter()
                self.advance_runners()
        elif reward > 1:
            print("reward was an out")
            self._current_total_strikes += 1
            # ball in play, but out
            self._current_outs += 1
            if self._current_outs >= 3:
                self.next_inning()
        else:
            print("reward was a strike")
            # strike
            self._current_total_strikes += 1
            self._current_strikes += 1
            if self._current_strikes >= 3:
                self._current_outs += 1
                if self._current_outs >= 3:
                    self.next_inning()
                self.next_batter()
                self.advance_runners()
            pass

    def next_batter(self):
        self._current_balls = 0
        self._current_strikes = 0
        self._current_batter += 1
        if self._current_batter >= len(self._batters):
            self._current_batter = 0

    def advance_runners(self):
        if self._current_runner_on_third:
            self._current_runner_on_third = 0
            self._current_runs += 1
        if self._current_runner_on_second:
            self._current_runner_on_second = 0
            self._current_runner_on_third = 1
        if self._current_runner_on_first:
            self._current_runner_on_second = 1
        self._current_runner_on_first = 1

    def next_inning(self):
        self._current_inning += 1
        self._current_outs = 0

    def print_full(self):
        print("")
        print("current pitch count", self._current_pitch_count)
        print("current total strikes thrown", self._current_total_strikes)
        print("current total balls thrown", self._current_total_balls)
        print("current inning", self._current_inning)
        print("current runs", self._current_runs)
        print("current outs", self._current_outs)
        print("current balls", self._current_balls)
        print("current strikes", self._current_strikes)
        print("current runner on first", self._current_runner_on_first)
        print("current runner on second", self._current_runner_on_second)
        print("current runner on third", self._current_runner_on_third)
        print("current batter", self._current_batter)
        print("current pitcher", self._pitcher_skill)

    def print_count(self):
        print(
            "inning",
            self._current_inning + 1,
            "balls",
            self._current_balls,
            "strikes",
            self._current_strikes,
            "outs",
            self._current_outs,
            "runs",
            self._current_runs,
        )

    def action_description(self, action):
        pitch_was_a_ball = action % len(self._pitch_intents) == 0
        if pitch_was_a_ball:
            pitch_desc = " - ball"
        else:
            pitch_desc = " - strike"
        pitch = int(action) // len(self._pitch_intents)
        if pitch == 0:
            return f"Fastball{pitch_desc}"
        elif pitch == 1:
            return f"Curveball{pitch_desc}"
        elif pitch == 2:
            return f"Slider{pitch_desc}"
        elif pitch == 3:
            return f"Changeup{pitch_desc}"
        elif pitch == 4:
            return f"Knuckleball{pitch_desc}"
        elif pitch == 5:
            return f"Splitter{pitch_desc}"
        elif pitch == 6:
            return f"Sinker{pitch_desc}"
        elif pitch == 7:
            return f"Cutter{pitch_desc}"
        else:
            return "Unknown"
