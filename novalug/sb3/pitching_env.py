import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random


class PitchingEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, pitcher, batters, defense_skill, innings=9):
        super().__init__()
        # print('in init')
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        actions = [
            0,  # "Fastball",
            1,  # "Curveball",
            2,  # "Slider",
            3,  # "Changeup",
            4,  # "Knuckleball",
            5,  # "Splitter",
            6,  # "Sinker",
            7,  # "Cutter",
        ]
        pitch_intent = [
            0,  # "Strike",
            1,  # "Ball",
        ]
        N_DISCRETE_ACTIONS = len(actions) * len(pitch_intent)
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # max values for the observation space
        pitches = 999
        runs = 999
        outs = 3
        pitcher_fb_skill = 10  # pitcher[0]
        pitcher_cb_skill = 10  # pitcher[1]
        pitcher_sl_skill = 10  # pitcher[2]
        pitcher_ch_skill = 10  # pitcher[3]
        pitcher_kn_skill = 10  # pitcher[4]
        pitcher_sp_skill = 10  # pitcher[5]
        pitcher_si_skill = 10  # pitcher[6]
        pitcher_cu_skill = 10  # pitcher[7]
        batters_count = len(batters)
        batter_1_skill = 10  # batters[0]
        batter_2_skill = 10  # batters[1]
        batter_3_skill = 10  # batters[2]
        batter_4_skill = 10  # batters[3]
        batter_5_skill = 10  # batters[4]
        batter_6_skill = 10  # batters[5]
        batter_7_skill = 10  # batters[6]
        batter_8_skill = 10  # batters[7]
        batter_9_skill = 10  # batters[8]
        balls = 4
        strikes = 3
        runner_on_first = 2
        runner_on_second = 2
        runner_on_third = 2
        discrete_array = [
            innings,
            runs,
            outs,
            pitches,
            pitcher_fb_skill,
            pitcher_cb_skill,
            pitcher_sl_skill,
            pitcher_ch_skill,
            pitcher_kn_skill,
            pitcher_sp_skill,
            pitcher_si_skill,
            pitcher_cu_skill,
            batters_count,
            batter_1_skill,
            batter_2_skill,
            batter_3_skill,
            batter_4_skill,
            batter_5_skill,
            batter_6_skill,
            batter_7_skill,
            batter_8_skill,
            batter_9_skill,
            balls,
            strikes,
            runner_on_first,
            runner_on_second,
            runner_on_third,
        ]
        # print(discrete_array)
        self.observation_space = spaces.MultiDiscrete(discrete_array)

        self._innings = innings
        self._batters = batters
        self._pitcher = pitcher
        self._defense_skill = defense_skill
        self.set_initial_values()

    def step(self, action):
        # print('in step')
        # self.render()
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
        # print('in reset')
        self.set_initial_values()
        observation = self.get_obs()
        # print('RESET observation', observation)
        info = self.get_info()
        # print('RESET info', info)
        return observation, info

    def render(self):
        # print('in render')
        self.print_count()

    def close(self):
        print("in close")
        pass

    # our stuff

    def get_obs(self):
        obs = [
            self._current_inning,
            self._current_runs,
            self._current_outs,
            self._current_pitch_count,
            self._pitcher[0],
            self._pitcher[1],
            self._pitcher[2],
            self._pitcher[3],
            self._pitcher[4],
            self._pitcher[5],
            self._pitcher[6],
            self._pitcher[7],
            # len(self._batters),
            self._current_batter,
            self._batters[0],
            self._batters[1],
            self._batters[2],
            self._batters[3],
            self._batters[4],
            self._batters[5],
            self._batters[6],
            self._batters[7],
            self._batters[8],
            self._current_balls,
            self._current_strikes,
            self._current_runner_on_first,
            self._current_runner_on_second,
            self._current_runner_on_third,
        ]
        # print(obs)
        return np.array(obs)

    def get_info(self):
        return {}

    def set_initial_values(self):
        self._current_pitch_count = 0
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
        # print('pitcher threw', self.action_description(action))
        # print("pitcher threw:", action)
        pitcher_tried_to_throw_a_strike = action % 2 == 0
        # print(type(action))
        if action <= 0:
            pitcher_skill_index = 0
        else:
            pitcher_skill_index = int(action) // 2
        pitcher_skill_at_pitch = self._pitcher[pitcher_skill_index]
        if pitcher_tried_to_throw_a_strike:
            strike_chances = [1] * pitcher_skill_at_pitch
            ball_chances = [0] * (10 - pitcher_skill_at_pitch)
        else:
            strike_chances = [1] * (10 - pitcher_skill_at_pitch)
            ball_chances = [0] * pitcher_skill_at_pitch

        # determine the pitch thrown
        # determine the pitcher's skill at that pitch
        # determine the batter's skill
        batter_skill = self._batters[self._current_batter]
        hit_chances = [-1] * batter_skill

        defense_skill = self._defense_skill
        out_chances = [2] * defense_skill

        chances = hit_chances + ball_chances + strike_chances + out_chances
        # print("chances", chances)
        reward = random.choice(chances)
        # print("reward", reward)
        return reward

        # print("pitch was a strike?", pitch_was_a_strike)
        # -1 is a hit
        # 0 is a ball
        # 1 is a strike
        # 2 is an out on a ball in play
        # return random.randint(-1, 2)

    def update_state(self, reward):
        self._current_pitch_count += 1
        if reward < 0:
            print("reward was a hit")
            # hit
            self.advance_runners()
            self.next_batter()
        elif reward == 0:
            print("reward was a ball")
            # ball
            self._current_balls += 1
            if self._current_balls >= 4:
                self.next_batter()
                self.advance_runners()
        elif reward > 1:
            print("reward was an out")
            # ball in play, but out
            self._current_outs += 1
            if self._current_outs >= 3:
                self.next_inning()
        else:
            print("reward was a strike")
            # strike
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
        # self._current_batter = self.batters[self.batters.index(self._current_batter) + 1]

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
        print("current pitch count", self._current_pitch_count)
        print("current inning", self._current_inning)
        print("current runs", self._current_runs)
        print("current outs", self._current_outs)
        print("current balls", self._current_balls)
        print("current strikes", self._current_strikes)
        print("current runner on first", self._current_runner_on_first)
        print("current runner on second", self._current_runner_on_second)
        print("current runner on third", self._current_runner_on_third)
        print("current batter", self._current_batter)
        print("current pitcher", self._pitcher)

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
