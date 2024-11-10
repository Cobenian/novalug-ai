from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent

import asyncio


from llama_index.llms.anthropic import Anthropic
from llama_index.core.prompts import PromptTemplate
from llama_index.core.bridge.pydantic import BaseModel

from stable_baselines3 import PPO

# from stable_baselines3.common.env_checker import check_env

from novalug.sb3.pitching_env import PitchingEnv
from novalug.li.game_state import GameState
from novalug.li.baseball_rules import BaseballRules


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


class ThrowAPitchEvent(Event):
    game_state: GameState


class PitchingFlow(Workflow):
    llm = Anthropic()

    @step
    async def start_game(self, ev: StartEvent) -> ChoosePitcherEvent:
        game_state = ev.game_state

        return ChoosePitcherEvent(game_state=game_state)

    @step
    async def choose_available_pitcher(
        self, ev: ChoosePitcherEvent
    ) -> ThrowAPitchEvent:
        game_state = ev.game_state

        pitchers = game_state.get_remaining_pitchers()

        # Format the list of pitchers into a string
        formatted_pitchers = format_pitchers(pitchers)

        prompt = f"Given the information about the following pitchers, tell me which pitcher should pitch next:\n{formatted_pitchers}\n\nPlease provide the index of the chosen pitcher. Explain your reasoning. At the end of your explanation, on a new line type the index of the chosen pitcher and ONLY the index without any other text."
        response = await self.llm.acomplete(prompt)
        print(response)

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

        print(f"Chosen pitcher: {chosen_pitcher}")

        pitcher_skills = chosen_pitcher.get_pitcher_skills()
        game_state.set_pitcher_skills(pitcher_skills)
        game_state.set_remaining_pitchers(remaining_pitchers)

        return ThrowAPitchEvent(game_state=game_state)

    @step
    async def throw_a_pitch(self, ev: ThrowAPitchEvent) -> InputRequiredEvent:
        game_state = ev.game_state

        env = PitchingEnv(game_state)

        model_file = "models/pitching_model_undertrained.zip"

        model = PPO.load(model_file, env=env)

        obs = env.get_obs()
        action, _states = model.predict(obs, deterministic=True)
        pitch = game_state.action_description(action)
        print("INTEND TO THROW PITCH:", pitch)

        return InputRequiredEvent(
            prefix="What happened after the pitch?", game_state=game_state
        )

    @step
    async def decide_if_should_change_pitchers(
        self, ev: HumanResponseEvent
    ) -> ChangePitcherDecisionEvent:
        game_state = ev.game_state
        print(ev.response)

        game_state.print_full()
        description_of_the_play = ev.response
        prompt = f"You are to choose between one of the following outcomes based on the text description provided. You can only choose one of hit, out, ball strike or game over. The description of the play was {description_of_the_play}. Resond with your reasoning. On the last line type only your choice with nothing else."
        response = await self.llm.acomplete(prompt)
        print(response)
        response_text = response.text.strip()
        last_line = response_text.split("\n")[-1]
        play = last_line.lower()
        print(play)
        if play in ["game over"]:
            return GameOverEvent(game_state=game_state)
        else:
            game_state.update_game_state_for_play(play)
            game_state.print_full()

        baseball_rules = BaseballRules()

        if game_state.game_is_over(baseball_rules):
            return GameOverEvent(game_state=game_state)

        return ChangePitcherDecisionEvent(game_state=game_state)

    @step
    async def change_pitcher(self, ev: ChangePitcherDecisionEvent) -> GameOverEvent:
        game_state = ev.game_state
        current_pitcher_stats = game_state.get_pitcher_stats()
        print(f"Current pitcher stats: {current_pitcher_stats}")

        prompt = f"Pitchers should not throw over 12 pitches or 50% balls. If they do or if they give up 3 runs they should be pulled from the game. Given these current pitcher stats: {current_pitcher_stats}, should we change the pitcher? Provide an explanation of your reasoning. On the last line, answer with yes or no. Do not provide any additional information on the last line."
        response = await self.llm.acomplete(prompt)
        print(response)
        response_text = response.text.strip()
        last_line = response_text.split("\n")[-1]
        should_change_pitcher = last_line.lower() == "yes"
        if should_change_pitcher:
            return ChoosePitcherEvent(game_state=game_state)
        else:
            return ThrowAPitchEvent(game_state=game_state)

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


async def main():
    workflow = PitchingFlow(timeout=60, verbose=False)

    handler = workflow.run(game_state=initial_game_state)
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            # here, we can handle human input however you want
            # this means using input(), websockets, accessing async state, etc.
            # here, we just use input()
            response = input(event.prefix)
            handler.ctx.send_event(
                HumanResponseEvent(response=response, game_state=event.game_state)
            )

    result = await handler
    print(str(result))


asyncio.run(main())
