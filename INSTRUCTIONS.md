## NOTE

Some of these instructions are out of date as I deleted some of the test files in the repository.

The purpose of this is to document the steps taken to add the dependencies. 

You should use poetry to install the dependencies.

You should follow the instructions in the README file to run the demo.

## Setup

```bash
brew install pipx
# you may need to source ~/.zshrc here
pipx ensurepath

pipx install poetry
poetry completions zsh > ~/.zfunc/_poetry

poetry new novalug
cd novalug
```

## XGBoost

```bash
poetry add xgboost
poetry add scikit-learn
```

Create the `novalug/hello_xgboost.py` file (with content)

```bash
poetry run python novalug/xgb/hello_xgboost.py
```

Error

```
xgboost.core.XGBoostError: 
XGBoost Library (libxgboost.dylib) could not be loaded.
Likely causes:
  * OpenMP runtime is not installed
    - vcomp140.dll or libgomp-1.dll for Windows
    - libomp.dylib for Mac OSX
    - libgomp.so for Linux and other UNIX-like OSes
    Mac OSX users: Run `brew install libomp` to install OpenMP runtime.

  * You are running 32-bit Python on a 64-bit OS
```

Fix

```bash
brew install libomp
```

Run again

```bash
poetry run python novalug/xgb/hello_xgboost.py
```

```bash
poetry add black
poetry run black .
```

Create `novalug/baseball.py` file with content

Add this to `pyproject.toml`

```
[tool.poetry.scripts]
baseball-entrypoint = "novalug.baseball:enter"
```

```bash
poetry run baseball-entrypoint
```

You will get a warning

```
Warning: 'baseball-entrypoint' is an entry point defined in pyproject.toml, but it's not installed as a script. You may get improper `sys.argv[0]`.

The support to run uninstalled scripts will be removed in a future release.

Run `poetry install` to resolve and get rid of this message.
```

```bash
poetry install

poetry run baseball-entrypoint
```


```bash
poetry add pandas

poetry run python novalug/xgb/hitting.py
```

## Streamlit

```bash
poetry add streamlit

poetry run streamlit run novalug/sl/site.py
```

Got an error with Streamlist and python 3.13, so:

```bash
poetry env use 3.12
poetry add streamlit black pandas scikit-learn xgboost
poetry update package
```

Useful website:

https://docs.streamlit.io/develop/quick-reference/cheat-sheet

```bash
poetry run streamlit run novalug/sl/site.py
```


## Hugging Face

```bash
poetry add transformers datasets evaluate accelerate
poetry add torch
poetry add emoji
poetry run python novalug/hf/happy.py
```

```bash
poetry add termcolor
poetry run python novalug/hf/summary.py
```

```bash
poetry run python novalug/hf/good_game.py
```

```bash
poetry add sentence-transformers
```


## Anthropic

```bash
poetry add anthropic

poetry run python novalug/ant/anthropic.py
```

Ask it: 

```
what teams were in the 2019 world series?
```

Then

```
who had the most hits?
```

Then

```
get me the data as json
```

```
only the json
```

```
raw json only please
```

```
what year was this?
```

```
explain
```

```
thank you
```

## Stable Baselines3

```bash
poetry add stable-baselines3
```

Train our model
```bash
poetry run python novalug/sb3/learn_to_pitch.py
```

Use our model to make predictions

```bash
poetry run python novalug/sb3/pitch.py
```

## LlamaIndex

```bash
poetry add llama-index
poetry add llama_index.llms.anthropic
```

llama-index doesn't work with numpy 2.0+

```bash
poetry add numpy@<2.0.0
```

## RAG (With Weaviate)

Run weaviate server in docker container

```bash
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.27.2
```

Test client connectivity to the server
```bash
poetry add weaviate-client
poetry add llama-index-vector-stores-weaviate
poetry run python novalug/wvt/ready.py
```

```bash
poetry add llama-index-embeddings-huggingface
```


## Agents

```bash
poetry add llama_agents
```