# Novalug Demo

- Start weaviate docker container

```bash
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.27.2
```

- Set environment variable with Anthropic API Key

```bash
export ANTHROPIC_API_KEY='...'
```

## Backstory - New coach for a baseball team

Oh no, the old coach has moved out of town! You have taken over the team.

The players know their positions in the field, but the coach didn't leave behind a batting order.

Luckily we know the players on the team and their hitting stats from the previous season.

Let's calculate the OPS for each player and use the top 9 batters in our lineup!

```bash
poetry run make-lineup
```

### Tools Used

* [Pandas](https://pandas.pydata.org/)

> Pandas is a library for working with data. The data is stored in data frames (2-dimensional like a spreadsheet) and it is easy to manipulate columns and rows using this library.

* [scikit-learn](https://scikit-learn.org/stable/)

> scikit-learn is a very popular machine learning library in python.

* [XGBoost](https://xgboost.readthedocs.io/en/stable/#)

> XGBoost is a machine learning library that implements Gradient Boosting. It is particularly useful for predicting column values in tabular data (2-dimensional).

### What did we learn?

`Gradient Boosting` is a `machine learning` technique that works well for making predictions from tabular data. XGBoost makes predicions based on decision trees.

You can make predictions that are:

* `Classification` 

> A choice from a set of pre-defined categories. Useful for predicting something is a, b, or c.

* `Regression`

> A numeric value. Useful for predicting how strong the relationship is or how likely something is.

> [!NOTE]
> Our example is a regression example because we predicted the OPS, a numeric value.

> [!TIP]
> Collect data that includes the values you would like to predict. It is required for training your model.

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

### Tools Used

* [Numpy](https://numpy.org/)

> Python library for scientific computing. It has N-dimensional arrays and numerical computing functions.

* [Gymnasium](https://gymnasium.farama.org/)

> Provides an API for `reinforcement learning`. It provides some reference `environments` (typically video games) but we did not use any. The older library was called `gym`.

* [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/)

> A library of reinforcement learning algorithms. It uses `PyTorch`. We used the `Proximal Policy Optimization` (`PPO`) algorithm for `Deep Reinforcement Learning` (`DRL`).

* [PyTorch](https://pytorch.org/)

> Library for building `deep learning` models. Typically used for `computer vision` (aka `image recognition`) and `natural language processing` (`NLP`). 

### What did we learn?

Use `Deep Reinforcement Learning` to create models that learn which action to take based on a reward function. 

> [!TIP]
> Too much training can overfit your model. Insufficient training can underfit your model which means that the predictions it makes will not be very good.

## Our first game

Finally, it is game day. But oh no, one of our hitters can't make it to the game. Let's find the hitter on the bench that is the most similar so we can put them in the game!

```bash
poetry run recommend-similar-hitter
```

### Tools Used

* [Pandas](https://pandas.pydata.org/)

* [PyTorch](https://pytorch.org/)

* [Sentence Transformers](https://sbert.net/)

> Library for computing `embeddings` for text and images. It can also calculate similarity scores. It is useful for semantic search.

### What did we learn?

We compute `embeddings` for the various hitters. An embedding is a numerical representation of the data. Typically they are `vectors`.

We use `cosine similarity` (which is the cosine angle between vectors) to measure the similarity between two vectors. We store the similarities in a 2-dimensional `similarity matrix` (Pandas DataFrame). 

This allows us to rank the other hitters by how similar they are to the hitter we are looking at.



## Use AI to call pitches for us

Earlier we developed a DRL model to help determine what the best pitch was to throw given the current state of the game.

Let's use that now to tell us which pitches to call in game!

```bash
poetry run call-pitches
```

### Tools Used

* [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html)

* [LlamaIndex](https://www.llamaindex.ai/)

> Framework for building `Large Language Model` (`LLM`) applications

* [Anthropic](https://www.anthropic.com/)

> Cloud based AI with an API. Competitor to OpenAI and ChatGPT. We use it because it is much cheaper and sufficiently capable for our needs.

> [!IMPORTANT]
> This is the only commercial product we use in this project. You will need a paid account and an API key to run this demo.

### What did we learn?

`Agentic workflows` offer an excellent balance of AI based decision making with some structure for the overall framework of what should be happening.

> [!TIP]
> You can use function calls, API calls, AI models and much more as steps in your workflow.

> [!TIP]
> Do not make calls to LLM's that are too complicated or you will get results that will not be easy to handle. We recommend having the AI do things like choosing from a small discrete set of options or having the API parse human provided text to provide text that is simpler for models to understand.

## AI generated game recaps

Now that our game is over, let's look at some real world AI generated game recaps.

### Tools Used

`GameChanger` (a commercial product) provides a real world example of using AI to summarize baseball games into a game recap article. The text of the game recaps was copied into the `data/game_recaps` folder (with names modified).

## Summarize a game recap

```bash
poetry run summarize-game-recap
```

### Tools Used

* [HuggingFace](https://huggingface.co/)

> Hugging Face provides machine learning models and datasets. It is similar to GitHub for AI models and datasets.

* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

> `Transformers` are a `neural network` architecture that change an input sequence into an output sequence.
> HuggingFace provides an API for transformers that works with PyTorch, TensorFlow and JAX.

### What did we learn?

`Hugging Face` is a website with models, datasets and much more provided by the community. We can use existing models with our data or use existing datasets to train our own models. We can use models for all sorts of tasks such as image recognition, audio processing, and Natural Language Processing (NLP) like our `recommender` or `similarity` example.

`Summarization` is a common NLP task.

> [!TIP]
> There are many models and datasets on HuggingFace. Be sure to experiment with several models to see which one works best for you.

> [!TIP]
> New models and datasets are being added to HuggingFace all the time. Be sure to check the website for the latest models.

> [!TIP]
> We have found this web page to be very helpful when trying to understand the types of models that exist on HuggingFace. [Hugging Face Tasks](https://huggingface.co/tasks)

## Was the game recap positive or negative?

```bash
poetry run game-sentiment
```

### Tools Used

* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

### What did we learn?

Models exist to perform `sentiment analysis` on text.

## Let's chat about our season

```bash
poetry run chat-about-the-season
```

```bash
poetry run python novalug/wvt/query.py
```

### Tools Used

* [Weaviate](https://weaviate.io/)

> Vector database useful for `Retrieval-Augmented Generation` (`RAG`).
> Weaviate offers a paid cloud version or you can run the open source database locally.

* [LlamaIndex](https://www.llamaindex.ai/)

* [Anthropic](https://www.anthropic.com/)

* [HuggingFace Sentence Transformers](https://huggingface.co/docs/transformers/index)

### What did we learn?

We can augment a `Large Language Model` or `LLM` with our own data with `Retrieval-Augmented Generation (RAG)`!

Our data is stored in a `vector store database`.

> [!CAUTION]
> Consider carefully before sending any data to an third party API or cloud service.

> [!NOTE]
> We did not cover [ollama](https://ollama.com/), but it is possible to run LLMs locally. You could replace Anthropic with an open source LLM run in ollama.

## Some more chat with agents

This planning agent will attempt to figure out what steps are necessary to solve the question asked.

```bash
poetry run python novalug/li/agents.py
```

### Tools Used

* [Weaviate](https://weaviate.io/)

* [LlamaIndex](https://www.llamaindex.ai/)

* [Pandas](https://pandas.pydata.org/)

### What did we learn?

`Agents` are a very exciting idea. We expect big advancements in these library capabilities in the next few years.

> [!WARNING]
> Agents tend to be highly unreliable. While they are a very exciting concept, they frequently fail to run.

## Build a website to commemorate our season!

```bash
poetry run streamlit run novalug/sl/site.py
```

### Tools Used

* [Streamlit](https://streamlit.io/)

> Library for creating websites via a python API. Useful for presenting data frames, charts, etc. with your findings.

### What did we learn?

Streamlit makes it convenient to build websites with your `AI`/`ML` findings. We could have also used [Jupyter](https://jupyter.org/), [Dash](https://dash.plotly.com/), or [Gradio](https://www.gradio.app/). 