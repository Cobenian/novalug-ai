from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent
from llama_index.llms.anthropic import Anthropic
from llama_index.core.bridge.pydantic import BaseModel
from stable_baselines3 import PPO
import asyncio
from termcolor import cprint

# from stable_baselines3.common.env_checker import check_env

from novalug.sb3.pitching_env import PitchingEnv
from novalug.baseball.game_state import GameState
from novalug.baseball.baseball_rules import BaseballRules


class Pitcher(BaseModel):
    fb: int
    cb: int
    sl: int
    ch: int
    kn: int
    sp: int
    si: int
    cu: int

    def get_pitcher_skills(self):
        return [self.fb, self.cb, self.sl, self.ch, self.kn, self.sp, self.si, self.cu]

    def __repr__(self):
        return f"Pitcher(fb={self.fb}, cb={self.cb}, sl={self.sl}, ch={self.ch}, kn={self.kn}, sp={self.sp}, si={self.si}, cu={self.cu})"


def format_pitchers(pitchers):
    return "\n".join([str(pitcher) for pitcher in pitchers])


class GameOverEvent(Event):
    game_state: GameState


class ChangePitcherDecisionEvent(Event):
    game_state: GameState


class ChoosePitcherEvent(Event):
    game_state: GameState


class PickPitchToThrowEvent(Event):
    game_state: GameState


class PitchingFlow(Workflow):
    llm = Anthropic()

    @step
    async def start_game(self, ev: StartEvent) -> ChoosePitcherEvent:
        game_state = ev.game_state
        cprint("Starting the game, we need to choose the starting pitcher", "yellow")
        return ChoosePitcherEvent(game_state=game_state)

    @step
    async def choose_available_pitcher(
        self, ev: ChoosePitcherEvent
    ) -> PickPitchToThrowEvent | GameOverEvent:
        game_state = ev.game_state

        pitchers = game_state.get_remaining_pitchers()

        if len(pitchers) == 0:
            cprint("No more pitchers available, will forfeit the game", "blue")
            return GameOverEvent(game_state=game_state)

        # Format the list of pitchers into a string
        formatted_pitchers = format_pitchers(pitchers)

        prompt = f"Given the information about the following pitchers, tell me which pitcher should pitch next:\n{formatted_pitchers}\n\nPlease provide the index of the chosen pitcher. Explain your reasoning. At the end of your explanation, on a new line type the index of the chosen pitcher and ONLY the index without any other text."
        response = await self.llm.acomplete(prompt)
        cprint(response, "yellow")

        # Access the text value of the completion response
        response_text = response.text.strip()

        # get the last line of the response
        last_line = response_text.split("\n")[-1]

        # Parse the response to get the chosen pitcher index
        chosen_pitcher_index = int(last_line)

        # Get the chosen pitcher object
        chosen_pitcher = pitchers[chosen_pitcher_index]

        remaining_pitchers = (
            pitchers[:chosen_pitcher_index] + pitchers[chosen_pitcher_index + 1 :]
        )

        cprint(f"Chosen pitcher: {chosen_pitcher}", "blue")

        pitcher_skills = chosen_pitcher.get_pitcher_skills()
        game_state.set_pitcher_skills(pitcher_skills)
        game_state.set_remaining_pitchers(remaining_pitchers)

        return PickPitchToThrowEvent(game_state=game_state)

    @step
    async def decide_which_pitch_to_throw(
        self, ev: PickPitchToThrowEvent
    ) -> InputRequiredEvent:
        game_state = ev.game_state

        env = PitchingEnv(game_state)

        model_file = "models/sb3/pitching_model_trained.zip"

        model = PPO.load(model_file, env=env)

        obs = env.get_obs()
        action, _states = model.predict(obs, deterministic=True)
        pitch = game_state.action_description(action)
        cprint(f"PITCH RELAYED TO THE PITCHER AND CATCHER: {pitch}", "blue")

        return InputRequiredEvent(
            prefix="What happened as a result of the pitch?", game_state=game_state
        )

    @step
    async def update_game_state_based_on_pitch_result(
        self, ev: HumanResponseEvent
    ) -> ChangePitcherDecisionEvent | GameOverEvent:
        game_state = ev.game_state
        # cprint(ev.response, 'yellow')

        # game_state.print_full()
        description_of_the_play = ev.response
        prompt = f"You are to choose between one of the following outcomes based on the text description provided. You can only choose one of hit, out, ball, strike or game over. The description of the play was {description_of_the_play}. Resond with your reasoning. On the last line type only your choice with nothing else."
        response = await self.llm.acomplete(prompt)
        cprint(response, "yellow")
        response_text = response.text.strip()
        last_line = response_text.split("\n")[-1]
        play = last_line.strip().lower()
        # cprint(play, "blue")
        if play not in ["hit", "out", "ball", "strike", "game over"]:
            cprint(
                "Invalid play, must be one of hit, out, ball, strike, game over", "red"
            )
            return PickPitchToThrowEvent(game_state=game_state)
        if play in ["game over"]:
            return GameOverEvent(game_state=game_state)
        else:
            game_state.update_game_state_for_play(play)
            cprint(game_state.get_current_inning_info(), "blue")

        baseball_rules = BaseballRules()

        if game_state.game_is_over(baseball_rules):
            return GameOverEvent(game_state=game_state)

        return ChangePitcherDecisionEvent(game_state=game_state)

    @step
    async def decide_if_we_should_change_pitcher(
        self, ev: ChangePitcherDecisionEvent
    ) -> ChoosePitcherEvent | PickPitchToThrowEvent:
        game_state = ev.game_state
        current_pitcher_stats = game_state.get_pitcher_stats()
        cprint(
            f"Should we change pitcher based on current pitcher stats: {current_pitcher_stats}",
            "green",
        )

        prompt = f"Pitchers must not throw less than 10 pitches. Pitchers should not throw more than 45 pitches. If the pitcher throws more than 50% balls after throwing 10 pitches you should pull them. If the pitcher gives up 3 runs they should be pulled from the game. Given these current pitcher stats: {current_pitcher_stats}, should we change the pitcher? Provide an explanation of your reasoning. On the last line, answer with yes or no. Do not provide any additional information on the last line."
        response = await self.llm.acomplete(prompt)
        cprint(response, "yellow")
        response_text = response.text.strip()
        last_line = response_text.split("\n")[-1]
        should_change_pitcher = last_line.lower() == "yes"
        if should_change_pitcher:
            return ChoosePitcherEvent(game_state=game_state)
        else:
            return PickPitchToThrowEvent(game_state=game_state)

    @step
    async def end_the_game(self, ev: GameOverEvent) -> StopEvent:
        game_state = ev.game_state
        return StopEvent(result=str("game over"))


pitchers = [
    Pitcher(fb=8, cb=3, sl=7, ch=0, kn=0, sp=0, si=0, cu=0),
    Pitcher(fb=7, cb=4, sl=6, ch=0, kn=0, sp=0, si=0, cu=0),
    Pitcher(fb=6, cb=5, sl=5, ch=0, kn=0, sp=0, si=0, cu=0),
    Pitcher(fb=5, cb=6, sl=4, ch=0, kn=0, sp=0, si=0, cu=0),
]

pitcher_skill = [5] * len(pitchers)
pitch_intents = [0, 0, 1]
batters = [8, 9, 7, 7, 4, 5, 3, 4, 4]
defense_skill = 7

initial_game_state = GameState(
    pitcher_skill, pitch_intents, batters, defense_skill, remaining_pitchers=pitchers
)
initial_game_state.set_initial_values()


async def run_workflow():
    workflow = PitchingFlow(timeout=None, verbose=False)

    handler = workflow.run(game_state=initial_game_state)
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            # here, we can handle human input however you want
            # this means using input(), websockets, accessing async state, etc.
            # here, we just use input()
            response = input(f"{event.prefix}\n")
            handler.ctx.send_event(
                HumanResponseEvent(response=response, game_state=event.game_state)
            )

    result = await handler
    cprint(str(result), "red")


def main():
    asyncio.run(run_workflow())
