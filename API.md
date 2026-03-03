# API Reference

All arrays are NumPy arrays.

## Result Types

- `SusieResult`
  - univariate fit container
  - key fields: `alpha (L,p)`, `mu (L,p)`, `mu2 (L,p)`, `pip (p,)`, `intercept (scalar)`

- `MultiSusieResult`
  - multivariate fit container (independent fit per outcome)
  - key fields: `fits: list[SusieResult]` length `r`, `intercept (r,)`

- `MixturePrior`
  - key fields: `weights (k,)`, `variances (k,)`, `mean_variance (scalar)`

## Fitting APIs

### `fit_susie_univariate(X, y, ...) -> SusieResult`

- `X`: `(n, p)`
- `y`: `(n,)`

### `fit_susie_multivariate_independent(X, Y, ...) -> MultiSusieResult`

- `X`: `(n, p)`
- `Y`: `(n, r)`

### `mvsusie(X, y_or_Y, ...) -> SusieResult | MultiSusieResult`

- If `y_or_Y` is `(n,)`, returns `SusieResult`.
- If `y_or_Y` is `(n, r)`, returns `MultiSusieResult`.
- Optional `mixture_prior`: if provided, uses `mixture_prior.mean_variance` as effective scalar prior variance.

### `mvsusie_suff_stat(XtX, Xty, yty, n, ...) -> SusieResult | MultiSusieResult`

- `XtX`: `(p, p)`
- Univariate mode:
  - `Xty`: `(p,)`
  - `yty`: scalar
- Multivariate mode:
  - `Xty`: `(p, r)`
  - `yty`: `(r,)`

### `mvsusie_rss(z, R, n, ...) -> SusieResult | MultiSusieResult`

- `R`: `(p, p)` LD/correlation-like matrix
- Univariate mode:
  - `z`: `(p,)`
- Multivariate mode:
  - `z`: `(p, r)`

### `mvsusie_rss_suff_stat(XtX, Xty, n, yty=None, ...) -> SusieResult | MultiSusieResult`

- Convenience wrapper over `mvsusie_suff_stat` for RSS-style sufficient stats.
- If `yty` is omitted, defaults to `n` (univariate) or `[n]*r` (multivariate).

## Prior API

### `create_mixture_prior(weights, variances, normalize_weights=True) -> MixturePrior`

- `weights`: `(k,)`, non-negative
- `variances`: `(k,)`, positive

## Utility APIs

### `coef(fit, include_intercept=False)`

- `fit` univariate:
  - `False` -> `(p,)`
  - `True` -> `(p+1,)` (`[intercept, beta...]`)
- `fit` multivariate:
  - `False` -> `(p, r)`
  - `True` -> `(p+1, r)` (intercept row on top)

### `predict(fit, X_new)`

- univariate -> `(n_new,)`
- multivariate -> `(n_new, r)`

### `calc_z(X, y) -> (p,)`

- `X`: `(n,p)`
- `y`: `(n,)`

### `mvsusie_single_effect_lfsr(fit)`

- univariate -> `(L,p)`
- multivariate -> `(L,p,r)`

### `mvsusie_get_lfsr(fit)`

- univariate -> `(p,)`
- multivariate -> `(p,r)`

## Example: Univariate

```python
import numpy as np
from mvsusie_py import mvsusie, coef, predict

rng = np.random.default_rng(1)
X = rng.normal(size=(200, 100))
b = np.zeros(100); b[5] = 3.0
y = -1 + X @ b + rng.normal(size=200)

fit = mvsusie(X, y, L=8, residual_variance=1.0, estimate_residual_variance=False)
print(coef(fit, include_intercept=True).shape)  # (101,)
print(predict(fit, X).shape)                    # (200,)
```

## Example: Multivariate

```python
import numpy as np
from mvsusie_py import mvsusie, coef, predict

rng = np.random.default_rng(2)
X = rng.normal(size=(150, 60))
B = np.zeros((60, 3)); B[3,0] = 2.0; B[8,1] = -1.8; B[12,2] = 1.5
Y = np.array([-1.0, 1.0, -2.0])[None, :] + X @ B + rng.normal(size=(150, 3))

fit = mvsusie(X, Y, L=8, residual_variance=np.ones(3), estimate_residual_variance=False)
print(coef(fit).shape)               # (60, 3)
print(coef(fit, True).shape)         # (61, 3)
print(predict(fit, X).shape)         # (150, 3)
```
