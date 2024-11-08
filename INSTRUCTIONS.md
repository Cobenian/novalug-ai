## Setup

```bash
brew install pipx
# you may need to source ~/.zshrc here
pipx ensurepath

pipx install poetry
poetry completions zsh > ~/.zfunc/_poetry

poetry new novalug
cd novalug
poetry add xgboost
poetry add scikit-learn
```

Create the `novalug/hello_xgboost.py` file (with content)

```bash
poetry run python novalug/hello_xgboost.py
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
poetry run python novalug/hello_xgboost.py
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