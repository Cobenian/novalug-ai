import os
from anthropic import Anthropic
from termcolor import cprint

client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)


def ask_claude(messages):
    message = client.messages.create(
        max_tokens=1024,
        messages=messages,
        model="claude-3-opus-20240229",
    )
    return message


if __name__ == "__main__":
    conversation_history = []

    while True:
        user_input = input("Ask Claude a question (or type 'quit' to exit): \n")
        if user_input.lower() == "quit":
            break

        # Append the user's message to the conversation history
        conversation_history.append({"role": "user", "content": user_input})

        # Get the response from Claude
        response = ask_claude(conversation_history)

        # Append Claude's response to the conversation history
        conversation_history.append({"role": "assistant", "content": response.content})

        # print(f"Claude's response: {response['content']}")
        # print(f"Claude's response: {response}")
        for line in response.content:
            cprint(line.text, "blue")
