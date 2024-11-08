import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random


class PitchingEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, pitcher_skill, batters, defense_skill, innings=9):
        super().__init__()
        pitch_intent = [
            0,  # "Strike",
            1,  # "Ball",
        ]
        N_DISCRETE_ACTIONS = len(pitcher_skill) * len(pitch_intent)
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # max values for the observation space
        max_pitch_count = 999
        runs = 999
        outs = 3
        batters_count = len(batters)
        balls = 4
        strikes = 3
        runner_on_first = 2  # boolean
        runner_on_second = 2  # boolean
        runner_on_third = 2  # boolean
        pitch_skills = [10] * len(pitcher_skill)
        batters_skills = [10] * len(batters)
        discrete_array = (
            [innings, runs, outs, max_pitch_count]
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
        self.set_initial_values()

    def step(self, action):
        print("pitch thrown", self.action_description(action))
        reward = self.calculate_reward(action)
        self.update_state(reward)
        observation = self.get_obs()
        info = self.get_info()
        terminated = (
            self._current_runs >= 999
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
        pitcher_tried_to_throw_a_strike = action % 2 == 0
        if action <= 0:
            pitcher_skill_index = 0
        else:
            pitcher_skill_index = int(action) // 2
        pitcher_skill_at_pitch = self._pitcher_skill[pitcher_skill_index]
        batter_skill = self._batters[self._current_batter]
        defense_skill = self._defense_skill

        # -1 is a hit
        # 0 is a ball
        # 1 is a strike
        # 2 is an out on a ball in play

        if pitcher_tried_to_throw_a_strike:
            hit_chances = [-1] * batter_skill
            ball_chances = [0] * (10 - pitcher_skill_at_pitch)
            strike_chances = [1] * pitcher_skill_at_pitch
            out_chances = [2] * defense_skill
        else:
            hit_chances = [-1] * batter_skill
            # since a ball was thrown, the batter has a lower chance of hitting
            half_length = len(hit_chances) // 2
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
        if action == 0:
            return "Fastball - Ball"
        elif action == 1:
            return "Fastball - Strike"
        elif action == 2:
            return "Curveball - Ball"
        elif action == 3:
            return "Curveball - Strike"
        elif action == 4:
            return "Slider - Ball"
        elif action == 5:
            return "Slider - Strike"
        elif action == 6:
            return "Changeup - Ball"
        elif action == 7:
            return "Changeup - Strike"
        elif action == 8:
            return "Knuckleball - Ball"
        elif action == 9:
            return "Knuckleball - Strike"
        elif action == 10:
            return "Splitter - Ball"
        elif action == 11:
            return "Splitter - Strike"
        elif action == 12:
            return "Sinker - Ball"
        elif action == 13:
            return "Sinker - Strike"
        elif action == 14:
            return "Cutter - Ball"
        elif action == 15:
            return "Cutter - Strike"
        else:
            return "Unknown"
