from termcolor import cprint
import random


class GameState:
    def __init__(
        self,
        pitcher_skills,
        pitch_intents,
        batters,
        defense_skill,
        inning=0,
        runs=0,
        outs=0,
        balls=0,
        strikes=0,
        runner_on_first=0,
        runner_on_second=0,
        runner_on_third=0,
        remaining_pitchers=[],
        remaining_batters=[],
    ):
        self._pitcher_skills = pitcher_skills
        self._pitch_intents = pitch_intents
        self._batters = batters
        self._defense_skill = defense_skill
        self._inning = inning
        self._runs = runs
        self._outs = outs
        self._balls = balls
        self._strikes = strikes
        self._runner_on_first = runner_on_first
        self._runner_on_second = runner_on_second
        self._runner_on_third = runner_on_third
        self._remaining_pitchers = remaining_pitchers
        self._remaining_batters = remaining_batters

    def set_initial_values(self):
        # current pitcher
        self._current_pitch_count = 0
        self._current_total_strikes = 0
        self._current_total_balls = 0
        self._current_total_hits = 0
        self._current_total_walks = 0
        self._current_total_base_runners = 0
        # game
        self._current_inning = 0
        self._current_runs = 0
        self._current_outs = 0
        self._current_balls = 0
        self._current_strikes = 0
        self._current_runner_on_first = 0
        self._current_runner_on_second = 0
        self._current_runner_on_third = 0
        self._current_batter = 0

    def randomize_offense_and_defense(self):
        self._pitcher_skills = [random.randint(0, 9) for _ in self._pitcher_skills]
        self._batters = [random.randint(0, 9) for _ in self._batters]
        self._defense_skill = random.randint(0, 9)

    def game_is_over(self, baseball_rules):
        return (
            self.game_in_invalid_state(baseball_rules)
            # this prevents the game from continuing after the last inning
            # note that it does not take extra innings into account, it is naive and focused only on one team
            or self.get_current_inning() >= baseball_rules.number_of_innings
        )

    def game_in_invalid_state(self, baseball_rules):
        return (
            # this prevents and overflow of the range of allowed pitches
            self.get_current_pitch_count() >= baseball_rules.max_pitch_count
            # this prevents and overflow of the range of allowed runs
            or self.get_current_runs() >= baseball_rules.max_runs
        )

    def get_pitcher_skills(self):
        return self._pitcher_skills

    def get_pitch_intents(self):
        return self._pitch_intents

    def get_defense_skill(self):
        return self._defense_skill

    def get_current_inning(self):
        return self._current_inning

    def get_current_runs(self):
        return self._current_runs

    def get_current_outs(self):
        return self._current_outs

    def get_current_pitch_count(self):
        return self._current_pitch_count

    def get_current_batter(self):
        return self._current_batter

    def get_batters(self):
        return self._batters

    def get_current_balls(self):
        return self._current_balls

    def get_current_strikes(self):
        return self._current_strikes

    def get_current_runner_on_first(self):
        return self._current_runner_on_first

    def get_current_runner_on_second(self):
        return self._current_runner_on_second

    def get_current_runner_on_third(self):
        return self._current_runner_on_third

    def get_remaining_pitchers(self):
        return self._remaining_pitchers

    def get_pitcher_stats(self):
        return {
            "pitch_count": self._current_pitch_count,
            "total_strikes": self._current_total_strikes,
            "total_balls": self._current_total_balls,
            "total_hits": self._current_total_hits,
            "total_walks": self._current_total_walks,
            "total_base_runners": self._current_total_base_runners,
            "runs": self._current_runs,
        }

    def get_current_inning_info(self):
        return {
            "inning": self._current_inning + 1,
            "runs": self._current_runs,
            "outs": self._current_outs,
            "balls": self._current_balls,
            "strikes": self._current_strikes,
            "runner_on_first": self._current_runner_on_first,
            "runner_on_second": self._current_runner_on_second,
            "runner_on_third": self._current_runner_on_third,
        }

    def set_pitcher_skills(self, pitcher_skills):
        self._pitcher_skills = pitcher_skills

    def set_remaining_pitchers(self, remaining_pitchers):
        self._remaining_pitchers = remaining_pitchers

    # -1 is a pitch this pitcher doesn't throw
    # 0 is a hit
    # 1 is a ball
    # 2 is a strike
    # 3 is an out on a ball in play

    def result_of_play_was_a_pitch_this_pitcher_does_not_throw(self, pitch):
        return pitch == -1 or pitch == "non_pitch"

    def result_of_play_was_a_hit(self, pitch):
        return pitch == 0 or pitch == "hit"

    def result_of_play_was_a_ball(self, pitch):
        return pitch == 1 or pitch == "ball"

    def result_of_play_was_a_strike(self, pitch):
        return pitch == 2 or pitch == "strike"

    def result_of_play_was_an_out(self, pitch):
        return pitch == 3 or pitch == "out"

    def handle_a_hit(self):
        self._current_total_base_runners += 1
        self._current_total_hits += 1
        self.advance_runners()
        self.next_batter()

    def handle_an_out(self):
        self._current_outs += 1
        if self._current_outs >= 3:
            self.next_inning()
        self.next_batter()

    def handle_a_walk(self):
        self._current_total_walks += 1
        self._current_total_base_runners += 1
        self.advance_runners()
        self.next_batter()

    def handle_a_ball_in_play(self):
        self._current_total_strikes += 1

    def handle_a_strike(self):
        self._current_total_strikes += 1
        self._current_strikes += 1

    def handle_a_ball(self):
        self._current_total_balls += 1
        self._current_balls += 1

    def score_ball(self):
        self.handle_a_ball()
        if self._current_balls >= 4:
            self.handle_a_walk()
            print("\tresult was a walk")
            return "walk"
        print("\tresult was a ball")
        return "ball"

    def score_strike(self):
        self.handle_a_strike()
        if self._current_strikes >= 3:
            print("\tresult was a strikeout!")
            self.handle_an_out()
            return "out"
        else:
            print("\tresult was a strike")
            return "strike"

    def score_hit(self):
        print("\tresult was a ball in play for a hit")
        self.handle_a_ball_in_play()
        self.handle_a_hit()
        return "hit"

    def score_out(self):
        print("\tresult was a ball in play for an out")
        self.handle_a_ball_in_play()
        self.handle_an_out()
        return "out"

    def update_game_state_for_play(self, result_of_play):
        self._current_pitch_count += 1
        if self.result_of_play_was_a_pitch_this_pitcher_does_not_throw(result_of_play):
            if random.choice([True, False]):
                self.score_hit()
            else:
                self.score_ball()
            return "non_pitch"
        elif self.result_of_play_was_a_hit(result_of_play):
            return self.score_hit()
        elif self.result_of_play_was_a_ball(result_of_play):
            return self.score_ball()
        elif self.result_of_play_was_a_strike(result_of_play):
            return self.score_strike()
        elif self.result_of_play_was_an_out(result_of_play):
            return self.score_out()
        else:
            raise ValueError(f"Invalid result of play {result_of_play}")

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
        self.summary_print("pitcher", self._pitcher_skills)

    def summary_print(self, title, value, color="blue"):
        cprint(f"\t{title}: {value}", color)

    def get_current_count_info(self):
        return {
            "inning": self._current_inning + 1,
            "balls": self._current_total_balls,
            "strikes": self._current_total_strikes,
            "outs": self._current_outs,
            "runs": self._current_runs,
            "hits": self._current_total_hits,
            "runner_on_first": self._current_runner_on_first,
            "runner_on_second": self._current_runner_on_second,
            "runner_on_third": self._current_runner_on_third,
        }

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
