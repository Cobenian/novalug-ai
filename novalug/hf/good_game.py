from transformers import pipeline

# classifier = pipeline("sentiment-analysis")
classifier = pipeline(
    model="finiteautomata/bertweet-base-sentiment-analysis", device="cpu"
)


# Read the content of the file
with open("data/game_one_recap.txt", "r") as file:
    text = file.read()

# with open("data/game_two_recap.txt", "r") as file:
#     text = file.read()

# small text version
# results = classifier(text)

# BUT that doesn't work because the text is too long


# Function to split text into chunks
def split_text(text, max_length=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
            current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Split the text into chunks
chunks = split_text(text, max_length=512)

# Perform sentiment analysis on each chunk
results = []
for chunk in chunks:
    result = classifier(chunk)
    results.extend(result)


for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# Print the results
for result in results:
    label = result["label"]
    score = round(result["score"], 4)
    if label in ["POSITIVE", "POS"]:
        sentiment = "Positive"
    elif label in ["NEGATIVE", "NEG"]:
        sentiment = "Negative"
    elif label in ["NEUTRAL", "NEU"]:
        sentiment = "Neutral"
    else:
        sentiment = "Unknown"
    print(f"Sentiment: {sentiment}, Score: {score}")
