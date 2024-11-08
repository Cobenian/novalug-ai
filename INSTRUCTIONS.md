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