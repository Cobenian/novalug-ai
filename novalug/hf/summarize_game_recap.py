from transformers import pipeline
from termcolor import cprint
import os

# Set the environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_text_summarizer():
    text_summarizer = pipeline(
        "summarization", model="Falconsai/text_summarization", device="cpu"
    )
    return text_summarizer


def summarize_game_recap(classifier, game_recap_filename):
    # Read the content of the file
    with open(game_recap_filename, "r") as file:
        text = file.read()

    cprint("Text To Summarize:", "yellow")
    print("")
    cprint(text, "blue")
    print("")

    summary = classifier(text, max_length=100, min_length=50, do_sample=False)

    cprint("Summary:", "yellow")
    print("")
    cprint(summary[0]["summary_text"], "green")


def main():
    game_recap_filename = "data/game_recaps/game_recap_1.txt"
    summarizer = get_text_summarizer()
    summarize_game_recap(summarizer, game_recap_filename)
