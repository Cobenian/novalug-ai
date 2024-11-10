class BaseballRules:
    def __init__(self):
        self.max_pitch_count = 999
        self.max_runs = 999
        self.number_of_innings = 9
        self.outs_per_inning = 3
        self.balls_per_plate_appearance = 4
        self.strikes_per_plate_appearance = 3
        self.runner_on_first = 2  # two possible values (yes or no)
        self.runner_on_second = 2  # two possible values (yes or no)
        self.runner_on_third = 2  # two possible values (yes or no)
