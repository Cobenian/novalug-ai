import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from termcolor import cprint


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
        self._max_pitch_count = max_pitch_count
        self.set_initial_values()

    def step(self, action):
        print("\ttried to throw pitch:", self.action_description(action))
        result = self.calculate_reward(action)
        self.update_state(result)
        reward = result
        # reward = self._current_runs * -1
        observation = self.get_obs()
        info = self.get_info()
        terminated = (
            self._current_pitch_count >= self._max_pitch_count
            or self._current_runs >= self._max_runs
            or self._current_inning >= self._innings
            or (self._current_inning >= self._innings and self._current_outs >= 3)
        )
        truncated = (
            self._current_pitch_count >= self._max_pitch_count
            or self._current_runs >= self._max_runs
        )
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        cprint("START OF GAME!!!!", "blue")
        # self._pitcher_skill = [random.randint(0, 9) for _ in self._pitcher_skill]
        # self._batters = [random.randint(0, 9) for _ in self._batters]
        # self._defense_skill = random.randint(0, 9)
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
        self._current_total_hits = 0
        self._current_total_walks = 0
        self._current_total_base_runners = 0
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

        # -1 is a pitch this pitcher doesn't throw
        # 0 is a hit
        # 1 is a ball
        # 2 is a strike
        # 3 is an out on a ball in play

        if pitcher_skill_at_pitch == 0:
            return -1

        if pitcher_tried_to_throw_a_ball:
            hit_chances = [0] * batter_skill
            # since a ball was thrown, the batter has a lower chance of hitting
            half_length = len(hit_chances) // 4
            hit_chances = hit_chances[:half_length]
            ball_chances = [1] * pitcher_skill_at_pitch
            strike_chances = [2] * (10 - pitcher_skill_at_pitch)
            # out_chances = [3] * defense_skill
        else:
            hit_chances = [0] * batter_skill
            ball_chances = [1] * (10 - pitcher_skill_at_pitch)
            strike_chances = [2] * pitcher_skill_at_pitch
            # out_chances = [3] * defense_skill

        chances = hit_chances + ball_chances + strike_chances # + out_chances
        result = random.choice(chances)
        if result == 0:
            # ball was hit, but did the defense get an out?
            hit_chances = [0] * (10-defense_skill)
            out_chances = [3] * defense_skill
            random.choice(hit_chances + out_chances)
        else:
            return result

    def update_state(self, reward):
        self._current_pitch_count += 1
        if reward <= 0:
            self._current_total_strikes += 1
            self._current_total_base_runners += 1
            self._current_total_hits += 1
            print("\tresult was a ball in play for a hit")
            # hit
            self.advance_runners()
            self.next_batter()
        elif reward == 1:
            print("\tresult was a ball")
            # ball
            self._current_total_balls += 1
            self._current_balls += 1
            if self._current_balls >= 4:
                print("\twalk!")
                self._current_total_walks += 1
                self._current_total_base_runners += 1
                self.next_batter()
                self.advance_runners()
        elif reward == 3:
            print("\tresult was a ball in play for an out")
            self._current_total_strikes += 1
            # ball in play, but out
            self._current_outs += 1
            if self._current_outs >= 3:
                print("\tend of the inning")
                self.next_inning()
                self.next_batter()
        else:
            # reward == 2
            print("\tresult was a strike")
            # strike
            self._current_total_strikes += 1
            self._current_strikes += 1
            if self._current_strikes >= 3:
                print("\tstrikeout!")
                self._current_outs += 1
                if self._current_outs >= 3:
                    self.next_inning()
                self.next_batter()

    def next_batter(self):
        self._current_balls = 0
        self._current_strikes = 0
        self._current_batter += 1
        if self._current_batter >= len(self._batters):
            self._current_batter = 0

    def advance_runners(self):
        if self._current_runner_on_third:
            print("\tscoring a run!")
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
        self._current_runner_on_first = 0
        self._current_runner_on_second = 0
        self._current_runner_on_third = 0

    def print_full(self):
        print("")
        cprint("SUMMARY", "blue")
        self.summary_print("runs", self._current_runs, "red")
        self.summary_print("pitch count", self._current_pitch_count)
        self.summary_print("total strikes thrown", self._current_total_strikes)
        self.summary_print("total balls thrown", self._current_total_balls)
        self.summary_print("total hits", self._current_total_hits)
        self.summary_print("total walks", self._current_total_walks)
        self.summary_print("total base runners", self._current_total_base_runners)
        cprint("CURRENT GAME STATE", "blue")
        self.summary_print("runs", self._current_runs)
        self.summary_print("inning", self._current_inning)
        self.summary_print("outs", self._current_outs)
        self.summary_print("balls", self._current_balls)
        self.summary_print("strikes", self._current_strikes)
        self.summary_print("runner on first", self._current_runner_on_first == 1)
        self.summary_print("runner on second", self._current_runner_on_second == 1)
        self.summary_print("runner on third", self._current_runner_on_third == 1)
        self.summary_print("batter", self._current_batter)
        self.summary_print("pitcher", self._pitcher_skill)

    def summary_print(self, title, value, color="blue"):
        cprint(f"\t{title}: {value}", color)

    def print_count(self):
        content = [
            "inning",
            str(self._current_inning + 1),
            "balls",
            str(self._current_balls),
            "strikes",
            str(self._current_strikes),
            "outs",
            str(self._current_outs),
            "runs",
            str(self._current_runs),
            (
                "no runners on"
                if not (
                    self._current_runner_on_first == 1
                    or self._current_runner_on_second == 1
                    or self._current_runner_on_third == 1
                )
                else ""
            ),
            "runner on 1B" if self._current_runner_on_first == 1 else "",
            "runner on 2B" if self._current_runner_on_second == 1 else "",
            "runner on 3B" if self._current_runner_on_third == 1 else "",
        ]
        content_str = " ".join(content)
        cprint(content_str, "green")

    def action_description(self, action):
        pitch_was_a_ball = action % len(self._pitch_intents) == 0
        if pitch_was_a_ball:
            pitch_desc = " for a ball"
        else:
            pitch_desc = " for a strike"
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


class PitcherTrainingEnv(PitchingEnv):
    def reset(self, seed=None, options=None):
        cprint("START OF TRAINING!!!!", "blue")
        self._pitcher_skill = [random.randint(0, 9) for _ in self._pitcher_skill]
        self._batters = [random.randint(0, 9) for _ in self._batters]
        self._defense_skill = random.randint(0, 9)
        self.set_initial_values()
        observation = self.get_obs()
        info = self.get_info()
        return observation, info
