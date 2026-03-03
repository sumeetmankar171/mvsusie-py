# mvsusie-py

Python MVP port of core `mvsusieR` workflows with univariate and multivariate support (multivariate currently implemented as independent per-outcome fits).

## What you get

- Dense-data fitting: `mvsusie(X, y_or_Y, ...)`
- Sufficient-stat fitting: `mvsusie_suff_stat(XtX, Xty, yty, n, ...)`
- RSS fitting: `mvsusie_rss(z, R, n, ...)`
- RSS sufficient-stat wrapper: `mvsusie_rss_suff_stat(...)`
- Mixture-prior helper: `create_mixture_prior(...)`
- Core utilities: `coef`, `predict`, `calc_z`, `mvsusie_get_lfsr`, `mvsusie_single_effect_lfsr`

## Project status

- Current version: `0.2.0`
- Test status: local suite passing (`30 passed`)
- CI: GitHub Actions workflow included

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Quick usage

```python
import numpy as np
from mvsusie_py import mvsusie, coef, predict

rng = np.random.default_rng(1)
X = rng.normal(size=(200, 100))
b = np.zeros(100); b[5] = 3.0
y = -1 + X @ b + rng.normal(size=200)

fit = mvsusie(X, y, L=8, residual_variance=1.0, estimate_residual_variance=False)
print(coef(fit, include_intercept=True).shape)
print(predict(fit, X).shape)
```

## Validate install

```bash
source .venv/bin/activate
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/parity_smoke.py
```

## API map

- Model/result classes
  - `SusieResult`
  - `MultiSusieResult`
  - `MixturePrior`
- Fit APIs
  - `fit_susie_univariate`
  - `fit_susie_multivariate_independent`
  - `mvsusie`
  - `mvsusie_suff_stat`
  - `mvsusie_rss`
  - `mvsusie_rss_suff_stat`
- Utility APIs
  - `create_mixture_prior`
  - `coef`
  - `predict`
  - `calc_z`
  - `mvsusie_single_effect_lfsr`
  - `mvsusie_get_lfsr`

## Documentation

- Full API signatures, shape contracts, and examples: [API.md](./API.md)
- Step-by-step run instructions with full option explanations: [RUN_GUIDE.md](./RUN_GUIDE.md)
- Contribution process: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Release process: [RELEASE_CHECKLIST.md](./RELEASE_CHECKLIST.md)
- Version history: [CHANGELOG.md](./CHANGELOG.md)

## Design notes

- Core inference uses an IBSS-style iterative update.
- Suff-stat and RSS modes map into equivalent internal sufficient-stat updates.
- Multivariate support currently fits each outcome independently and stores them in `MultiSusieResult`.

## Limitations and roadmap

Current limitations:

- No full joint multivariate covariance inference yet.
- No mash-style covariance mixture learning yet.
- No dedicated CLI interface; Python API is the primary interface.

Roadmap direction:

1. Full multivariate coupled inference.
2. Richer prior learning and mash-like components.
3. Optional CLI wrapper for file-based workflows.

## License

MIT. See [LICENSE](./LICENSE).
