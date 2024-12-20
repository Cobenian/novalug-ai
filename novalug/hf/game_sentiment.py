from transformers import pipeline


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


def determine_game_sentiment(classifier, game_recap_filename):
    # Read the content of the file
    with open(game_recap_filename, "r") as file:
        text = file.read()

    # this would work if the text was small enough
    # results = classifier(text)
    # BUT that doesn't work because the text is too long
    # So we split the text into chunks
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


def main():
    game_recap_filename = "data/game_recaps/game_recap_1.txt"
    classifier = pipeline(
        model="finiteautomata/bertweet-base-sentiment-analysis", device="cpu"
    )
    determine_game_sentiment(classifier, game_recap_filename)
