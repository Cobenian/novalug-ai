from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
import asyncio


from llama_index.llms.anthropic import Anthropic
from llama_index.core.prompts import PromptTemplate
from llama_index.core.bridge.pydantic import BaseModel

from stable_baselines3 import PPO

# from stable_baselines3.common.env_checker import check_env

from novalug.sb3.pitching_env import PitchingEnv
from novalug.li.game_state import GameState

import numpy as np

print(np.__version__)


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

        print(f"Chosen pitcher: {chosen_pitcher}")

        pitcher_skills = chosen_pitcher.get_pitcher_skills()
        game_state.set_pitcher_skills(pitcher_skills)

        return ThrowAPitchEvent(game_state=game_state)

    @step
    async def throw_a_pitch(self, ev: ThrowAPitchEvent) -> StopEvent:
        response = "time to throw a pitch"

        env = PitchingEnv(game_state)

        model_file = "models/pitching_model_undertrained.zip"

        model = PPO.load(model_file, env=env)

        obs = env.get_obs()
        action, _states = model.predict(obs, deterministic=True)
        pitch = game_state.action_description(action)
        print("INTEND TO THROW PITCH:", pitch)

        return StopEvent(result=str(pitch))


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

game_state = GameState(
    pitcher_skill, pitch_intents, batters, defense_skill, remaining_pitchers=pitchers
)
game_state.set_initial_values()


async def main():
    w = PitchingFlow(timeout=60, verbose=False)
    result = await w.run(game_state=game_state)
    print(str(result))


asyncio.run(main())
