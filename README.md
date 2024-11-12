# Novalug Demo

- Start weaviate docker container

## Backstory - New coach for a baseball team

Oh no, the old coach has moved out of town! You have taken over the team.

The players know their positions in the field, but the coach didn't leave behind a batting order.

Luckily we know the players on the team and their hitting stats from the previous season.

Let's calculate the OPS for each player and use the top 9 batters in our lineup!

```bash
poetry run make-lineup
```

### What did we learn?

`Gradient Boosting` is a machine learning technique that works well for make predictions from tabular data. 

> [!TIP]
> Collect data that includes the values you would like to predict.

> [!TIP]
> Be prepared to clean your data and normalize your data if necessary.

## Learn To Call Pitches

I haven't coached before, so I am not good at calling pitches. What if AI could learn to call pitches for me?

First we need to train the AI to learn which pitches to call based on the game situation.

```bash
poetry run learn-to-pitch
```

Now let's use the model to simulate a game and see what it learned.

```bash
poetry run practice-calling-pitches
```

### What did we learn?

Use `Deep Reinforcement Learning` to create models that learn which action to take based on a reward function. 

> [!TIP]
> Too much training can overfit your model. Insufficient training can underfit your model which means that the predictions it makes will not be very good.

## Our first game

Finally, it is game day. But oh no, one of our hitters can't make it to the game. Let's find the hitter on the bench that is the most similar so we can put them in the game!

```bash
poetry run recommend-similar-hitter
```

### What did we learn?

`Hugging Face` is a website with models, datasets and much more provided by the community. We can use existing models with our data or use existing datasets to train our own models.

> [!TIP]
> There are many models and dataset on HuggingFace. Be sure to experiment with several models to see which one works best for you.

> [!TIP]
> New models and datasets are being added all the time. Be sure to check the website for the latest models.

> [!TIP]
> We have found this web page to be very helpful when trying to understand the types of models that exist on HuggingFace. [Hugging Face Tasks](https://huggingface.co/tasks)

## Use AI to call pitches for us

Earlier we developed a DRL model to help determine what the best pitch was to throw given the current state of the game.

Let's use that now to tell us which pitches to call in game!

```bash
poetry run call-pitches
```

### What did we learn?

Agentic workflows offer an excellent balance of AI based decision making with some structure for the overall framework of what should be happening.

> [!TIP]
> You can use function calls, API calls, AI models and much more as steps in your workflow.

> [!TIP]
> Do not make calls to LLM's that are too complicated or you will get results that will not be easy to handle. We recommend having the AI do things like choosing from a small discrete set of options or having the API parse human provided text to provide text that is simpler for models to understand.

## AI generated game recaps

Now that our game is over, let's look at some real world AI generated game recaps.

## Summarize a game recap

```bash
poetry run summarize-game-recap
```

## Was the game recap positive or negative?

```bash
poetry run game-sentiment
```

## Let's chat about our season

```bash
poetry run chat-about-the-season
```

```bash
poetry run python novalug/wvt/query.py
```

## Some more chat with agents

```bash
poetry run python novalug/li/agents.py
```