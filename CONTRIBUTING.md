# Contributing

Thank you for helping improve KobzarAI. Please follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Before you start

- Open an issue for substantial changes so the approach can be discussed.
- Keep changes focused and include tests for changed behavior.
- Do not commit credentials, local corpora, model weights, generated outputs,
  or content you do not have permission to contribute.
- Do not use holdout cases to tune models, prompts, retrieval, or thresholds.
  If a holdout is used for tuning, it is no longer an independent test.

## Development

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync --project ml
uv sync --project backend
uv run --project ml ruff check ml/evaluation backend
uv run --project ml python -m unittest discover -s ml/evaluation -p 'test_*.py'
uv run --project backend python -m unittest backend.test_main
```

Run these commands from the repository root. See the
[ML evaluation guide](ml/evaluation/README.md) and
[backend guide](backend/README.md) for local corpus and Ollama requirements.
Unit tests do not require a downloaded model.

## Pull requests

Describe the problem, your changes, and how they were tested. For model or
retrieval experiments, include the dataset split, relevant configuration,
limitations, and results; do not present development scores as independent
evaluation. Link related issues and document user-facing changes.
