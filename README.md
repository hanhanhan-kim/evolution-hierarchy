
# Evolutionary dynamic across hierarchy of phenotype

The manuscript (LaTeX source, bibliography, figures) lives in [`paper/`](paper/). Preprint: https://arxiv.org/abs/2203.09719

`paper/` is synced with Overleaf (remote `overleaf`) via git subtree:

```bash
git subtree pull --prefix=paper overleaf master -m "Pull from Overleaf"  # bring in Overleaf edits
git subtree push --prefix=paper overleaf master --rejoin                # send local edits to Overleaf (--rejoin keeps later pulls clean)
```

This guide will walk you through setting up and running this package using `uv` for package management and `marimo` for execution.

## Prerequisites

- Python 3.10 or higher
- `uv` package manager installed

## Installation Steps

1. Create a new virtual environment using `uv`:

```bash
uv venv
```

2. Activate the virtual environment:

- On Windows:

```bash
.venv\Scripts\activate
```

- On Unix/Linux/macOS:

```bash
source .venv/bin/activate
```

3. Install dependencies from the requirements file:

```bash
uv pip install -r requirements.txt
```

## Running simulation

Once the installation is complete, you can run the main simulation using marimo:

```bash
marimo edit main.py
```

This will open the notebook in your default web browser. You can then execute the cells to run the evolutionary algorithm.

## Troubleshooting

If you encounter any issues:

1. Ensure your virtual environment is activated (you should see `(.venv)` in your terminal prompt)
2. Verify all dependencies were installed correctly:

```bash
pip list
```

3. Check that Python version matches requirements:

```bash
python --version
```

## Deactivating the Environment

When you're done, you can deactivate the virtual environment:

```bash
uv venv deactivate
```
