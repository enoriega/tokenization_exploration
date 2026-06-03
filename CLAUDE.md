# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Exploration project for building and testing a custom BPE (Byte Pair Encoding) tokenizer. Training data is PubMed XML files (`data/` directory, gitignored).

## Package Manager

This project uses `uv`. Always use `uv` to run scripts and manage dependencies — not `pip` or `python` directly.

```bash
uv run python main.py          # Run a script
uv run jupyter notebook        # Start Jupyter
uv add <package>               # Add a dependency
uv sync                        # Install/sync dependencies
```

## Key Dependencies

- `tokenizers` (HuggingFace) — core BPE tokenizer implementation
- `transformers` (HuggingFace) — model/tokenizer loading utilities
- `jupyter` — notebooks for exploration and visualization

## Data

PubMed XML files live in `data/` (gitignored). The current dataset is `data/pubmed26n1330.xml`.
