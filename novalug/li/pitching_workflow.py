from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
import asyncio

# `pip install llama-index-llms-openai` if you don't already have it
from llama_index.llms.anthropic import Anthropic


class DeterminePitchEvent(Event):
    pitch: str


class ChoosePitcherEvent(Event):
    pitchers: list


class ThrowPitchEvent(Event):
    result: str


class UpdateGameStateEvent(Event):
    pass


class ChangePitcherEvent(Event):
    pass


class PitchingFlow(Workflow):
    llm = Anthropic()

    @step
    async def start_game(self, ev: StartEvent) -> ChoosePitcherEvent:
        pitchers = ev.pitchers

        # prompt = f"Which pitch should I throw next {topic}."
        # response = await self.llm.acomplete(prompt)
        return ChoosePitcherEvent(pitchers=pitchers)

    @step
    async def critique_joke(self, ev: ChoosePitcherEvent) -> StopEvent:
        pitchers = ev.pitchers

        prompt = f"Given the information about the following pitchers, tell me which pitcher should pitch next: {pitchers}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))


pitchers = [
    {"fb": 8, "cb": 3, "sl": 7, "ch": 0, "kn": 0, "sp": 0, "si": 0, "cu": 0},
    {"fb": 7, "cb": 4, "sl": 6, "ch": 0, "kn": 0, "sp": 0, "si": 0, "cu": 0},
    {"fb": 6, "cb": 5, "sl": 5, "ch": 0, "kn": 0, "sp": 0, "si": 0, "cu": 0},
    {"fb": 5, "cb": 6, "sl": 4, "ch": 0, "kn": 0, "sp": 0, "si": 0, "cu": 0},
]


async def main():
    w = PitchingFlow(timeout=60, verbose=False)
    result = await w.run(pitchers=pitchers)
    print(str(result))


asyncio.run(main())
