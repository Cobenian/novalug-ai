from transformers import pipeline
from termcolor import cprint
import os

# Set the environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

classifier = pipeline(
    "summarization", model="Falconsai/text_summarization", device="cpu"
)

# Read the content of the file
with open("data/game_one_recap.txt", "r") as file:
    text = file.read()

cprint("Text To Summarize:", "blue")
cprint(text, "blue")

summary = classifier(text, max_length=100, min_length=50, do_sample=False)

cprint("Summary:", "green")
cprint(summary[0]["summary_text"], "green")
