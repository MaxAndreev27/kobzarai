# KobzarAI

KobzarAI is an experimental, local Ukrainian-language AI companion inspired
by the works of Taras Shevchenko. It combines source retrieval with an Ollama
language model and a minimal FastAPI backend. The assistant is an AI character,
not the historical person.

The project is a research prototype, **not production-ready**. Retrieved
passages may be irrelevant or misattributed, and the model can invent facts
even when sources are provided. Responses require human verification. The API
has no authentication and must not be exposed to an untrusted network.

## Repository

- [`ml/`](ml/) — corpus preparation, retrieval experiments, evaluation, and tests.
- [`backend/`](backend/) — experimental single-turn local FastAPI service.
- `frontend/` — planned; not yet implemented.

The API currently supports explicit title-based lexical search and exact
normalized quote search. Hybrid retrieval and confidence thresholds remain
experiments; they are not used by the API.

## Getting started

You need Python 3.12, [uv](https://docs.astral.sh/uv/), and a local
[Ollama](https://ollama.com/) installation with `qwen3:8b`. See the
[backend guide](backend/README.md) for corpus setup, startup, an API example,
and known limitations.

Corpus data, downloaded models, and experiment outputs are not committed.
To build the corpus from the configured source, run these commands from the
repository root after checking the source's terms:

```sh
uv sync --project ml
PYTHONPATH=ml/src uv run --project ml python -m kobzarai.scrape
PYTHONPATH=ml/src uv run --project ml python -m kobzarai.filter
```

These commands use the network when source pages are not cached and overwrite
the corresponding processed corpus files. Follow the [backend guide](backend/README.md)
to create the separate quarantined copy used by the API. See the
[ML guide](ml/evaluation/README.md) for experiments, and check model and data
licenses before downloading or redistributing their content.

## Evaluation and contributing

The development results are limited and are not an independent human-reviewed
benchmark. See the [evaluation guide](ml/evaluation/README.md) and
[human review worksheet](ml/evaluation/HUMAN_REVIEW.md). Contributions are
welcome; please read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

Project code and documentation are available under the [MIT License](LICENSE).
Third-party texts, models, and datasets are subject to their own terms.
