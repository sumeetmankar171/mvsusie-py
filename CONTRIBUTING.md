# Contributing

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Run tests

```bash
source .venv/bin/activate
PYTHONPATH=src python -m pytest -q
```

## Coding guidelines

- Keep APIs shape-explicit (document expected array shapes).
- Add tests for every new public API or behavior change.
- Prefer small, composable functions for univariate and multivariate paths.
- Preserve deterministic test behavior with fixed NumPy random seeds.

## Pull requests

- Include a short summary of behavior changes.
- Add or update tests.
- Update `CHANGELOG.md` for user-facing changes.
- Ensure CI is green.
