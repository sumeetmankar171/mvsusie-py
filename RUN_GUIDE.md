# mvsusie-py Run Guide

This guide explains exactly how to run `mvsusie-py`, what each command does, and what each function option means.

## 1. What this package does

`mvsusie-py` is a Python MVP implementation of core `mvsusieR` workflows:

- Dense-data fitting (`mvsusie`)
- Sufficient-stat fitting (`mvsusie_suff_stat`)
- RSS fitting (`mvsusie_rss`)
- RSS sufficient-stat fitting (`mvsusie_rss_suff_stat`)
- Univariate and multivariate outcomes (multivariate currently via independent per-outcome fits)

## 2. Prerequisites

- Python `>=3.9`
- `pip`
- `numpy`
- `pytest` (for tests)

## 3. Step-by-step setup

From the project root:

```bash
cd ~/mvsusie-py
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Explanation:

- `python3 -m venv .venv`: creates isolated Python environment.
- `source .venv/bin/activate`: activates that environment.
- `pip install -e ".[dev]"`: editable install + dev deps (`pytest`).

## 4. Basic run commands

### 4.1 Run all tests

```bash
source .venv/bin/activate
PYTHONPATH=src python -m pytest -q
```

- `PYTHONPATH=src`: ensures local source package is importable.
- `-q`: quiet pytest output.

### 4.2 Run parity smoke check

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/parity_smoke.py
```

Expected:

- top PIP agreement across dense / suff-stat / rss in the synthetic test
- high coefficient correlation between dense and suff-stat

## 5. Minimal usage recipes

### 5.1 Univariate dense fit

```python
import numpy as np
from mvsusie_py import mvsusie, coef, predict

rng = np.random.default_rng(1)
n, p = 200, 100
X = rng.normal(size=(n, p))
b = np.zeros(p); b[5] = 3.0
y = -1.0 + X @ b + rng.normal(size=n)

fit = mvsusie(
    X, y,
    L=8,
    prior_variance=1.0,
    residual_variance=1.0,
    estimate_residual_variance=False,
    max_iter=200,
    tol=1e-6,
)

print(coef(fit).shape)                       # (p,)
print(coef(fit, include_intercept=True).shape)  # (p+1,)
print(predict(fit, X).shape)                 # (n,)
```

### 5.2 Multivariate dense fit

```python
import numpy as np
from mvsusie_py import mvsusie, coef, predict

rng = np.random.default_rng(2)
n, p, r = 150, 60, 3
X = rng.normal(size=(n, p))
B = np.zeros((p, r))
B[3,0], B[8,1], B[12,2] = 2.0, -1.8, 1.5
Y = np.array([-1.0, 1.0, -2.0])[None, :] + X @ B + rng.normal(size=(n, r))

fit = mvsusie(
    X, Y,
    L=8,
    residual_variance=np.ones(r),
    estimate_residual_variance=False,
)

print(coef(fit).shape)                   # (p, r)
print(coef(fit, include_intercept=True).shape)  # (p+1, r)
print(predict(fit, X).shape)             # (n, r)
```

### 5.3 Sufficient-stat fit

```python
import numpy as np
from mvsusie_py import mvsusie_suff_stat

# Assume centered Xc, yc
XtX = Xc.T @ Xc
Xty = Xc.T @ yc
yty = float(yc @ yc)
fit = mvsusie_suff_stat(XtX, Xty, yty, n=Xc.shape[0])
```

### 5.4 RSS fit

```python
import numpy as np
from mvsusie_py import mvsusie_rss

# z: marginal z-scores, R: LD/correlation-like matrix
fit = mvsusie_rss(z=z, R=R, n=n)
```

## 6. Detailed options and flags

There is no dedicated CLI parser in this package yet; the main "options" are function arguments.

### 6.1 `mvsusie(...)`

Signature:

```python
mvsusie(
  X,
  y,
  L=10,
  prior_variance=1.0,
  mixture_prior=None,
  residual_variance=None,
  estimate_residual_variance=True,
  max_iter=200,
  tol=1e-4,
)
```

Options:

- `X`: ndarray `(n,p)`
  - Dense design matrix.
- `y`: ndarray `(n,)` or `(n,r)`
  - Response vector or matrix.
- `L` (int, default `10`)
  - Number of single-effect components in SuSiE.
- `prior_variance` (float, default `1.0`)
  - Scalar prior variance for effect sizes.
- `mixture_prior` (`MixturePrior | None`)
  - If provided, package uses `mixture_prior.mean_variance` as effective scalar prior variance.
- `residual_variance` (float, vector, or `None`)
  - If `None`, initialized from variance of centered response.
  - For multivariate `y`, can pass one scalar or `(r,)` vector.
- `estimate_residual_variance` (bool, default `True`)
  - If `True`, updates residual variance each iteration.
- `max_iter` (int, default `200`)
  - Maximum IBSS iterations.
- `tol` (float, default `1e-4`)
  - Convergence tolerance on coefficient change.

Returns:

- Univariate: `SusieResult`
- Multivariate: `MultiSusieResult`

### 6.2 `mvsusie_suff_stat(...)`

Signature:

```python
mvsusie_suff_stat(
  XtX,
  Xty,
  yty,
  n,
  L=10,
  prior_variance=1.0,
  residual_variance=None,
  estimate_residual_variance=True,
  max_iter=200,
  tol=1e-4,
)
```

Options:

- `XtX`: `(p,p)`
- `Xty`: `(p,)` or `(p,r)`
- `yty`: scalar or `(r,)`
- `n`: sample size
- Remaining options match `mvsusie` semantics.

### 6.3 `mvsusie_rss(...)`

Signature:

```python
mvsusie_rss(
  z,
  R,
  n,
  L=10,
  prior_variance=1.0,
  residual_variance=1.0,
  estimate_residual_variance=False,
  max_iter=200,
  tol=1e-4,
)
```

Options:

- `z`: `(p,)` or `(p,r)` marginal z-scores.
- `R`: `(p,p)` LD/correlation-like matrix.
- `n`: GWAS sample size.
- `residual_variance`: scalar or `(r,)`.
- Other options same meaning as dense fit.

Internal approximation used:

- `z_tilde = z / sqrt(1 + z^2 / n)`
- `XtX = n * R`
- `Xty = sqrt(n) * z_tilde`
- `yty = n` (or vector of `n` in multivariate)

### 6.4 `mvsusie_rss_suff_stat(...)`

Signature:

```python
mvsusie_rss_suff_stat(
  XtX,
  Xty,
  n,
  yty=None,
  L=10,
  prior_variance=1.0,
  residual_variance=1.0,
  estimate_residual_variance=False,
  max_iter=200,
  tol=1e-4,
)
```

Options:

- `yty`: if `None`, defaults to `n` (or `[n]*r` for multivariate).
- Other options same semantics.

### 6.5 Utility options

`coef(fit, include_intercept=False)`

- `include_intercept=False`: only regression coefficients.
- `True`: prepend intercept (univariate) or intercept row (multivariate).

`predict(fit, X_new)`

- Uses model coefficients + intercept to predict on new design matrix.

`create_mixture_prior(weights, variances, normalize_weights=True)`

- `normalize_weights=True`: weights automatically normalized to sum to 1.
- `False`: raises error if weights do not already sum to 1.

## 7. Troubleshooting

### Problem: `ModuleNotFoundError: mvsusie_py`

Run:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

or:

```bash
PYTHONPATH=src python -m pytest -q
```

### Problem: no tests discovered

Make sure you are in repo root and run:

```bash
python -m pytest -q
```

### Problem: shape errors

Check these first:

- `X` rows must match `y` length or `Y` rows.
- `XtX` must be square.
- `Xty` first dimension must equal `XtX.shape[0]`.
- `z` first dimension must equal `R.shape[0]`.

## 8. Current limitations

- Multivariate path is independent per outcome (not full joint covariance model yet).
- No full mash-style covariance mixture inference yet.
- No dedicated end-user CLI yet; APIs are Python-first.

## 9. Recommended run workflow

1. Install editable package in a venv.
2. Run `pytest` to validate environment.
3. Start with dense univariate fit.
4. Move to multivariate or RSS workflows as needed.
5. Compare dense/suff-stat/rss behavior using `scripts/parity_smoke.py`.
