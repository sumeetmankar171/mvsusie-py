# API Reference

All arrays are NumPy arrays. Shape notation: `n` = samples, `p` = SNPs/predictors, `r` = traits/outcomes, `L` = single effects, `K` = mixture components.

---

## Result Types

### `SusieResult`
Univariate or per-outcome fit container.

| Field | Shape | Description |
|---|---|---|
| `alpha` | `(L, p)` | Posterior inclusion probabilities per effect |
| `mu` | `(L, p)` | Posterior means |
| `mu2` | `(L, p)` | Posterior second moments |
| `pip` | `(p,)` | Marginal posterior inclusion probabilities |
| `intercept` | scalar | Fitted intercept |
| `residual_variance` | scalar | Estimated σ² |
| `prior_variance` | scalar | Prior variance V |
| `converged` | bool | Whether IBSS converged |
| `n_iter` | int | Iterations run |

### `MultiSusieResult`
Independent per-outcome multivariate fit container.

| Field | Shape | Description |
|---|---|---|
| `fits` | `list[SusieResult]` length `r` | One fit per outcome |
| `intercept` | `(r,)` | Per-outcome intercepts |
| `prior_variance` | scalar | Shared prior variance |

### `JointMultiSusieResult`
Full joint multivariate fit with shared residual covariance.

| Field | Shape | Description |
|---|---|---|
| `alpha` | `(L, p)` | Shared posterior inclusion probabilities |
| `mu` | `(L, p, r)` | Posterior means per outcome |
| `mu2` | `(L, p, r)` | Posterior second moments per outcome |
| `pip` | `(p,)` | Marginal PIPs (shared across traits) |
| `intercept` | `(r,)` | Per-outcome intercepts |
| `residual_variance` | `(r, r)` | Learned joint residual covariance Σ |
| `prior_variance` | scalar | Prior variance V (learned if `estimate_prior_variance=True`) |
| `prior_variance_trace` | `list[float]` | V at each iteration (empty if not estimated) |
| `converged` | bool | Whether IBSS-M converged |
| `n_iter` | int | Iterations run |

### `MashCovariance`
Learned mash-style covariance mixture.

| Field | Shape | Description |
|---|---|---|
| `weights` | `(K,)` | EM-learned mixture weights (sum to 1) |
| `components` | `(K, r, r)` | Covariance matrices (canonical + data-driven) |
| `loglik_trace` | `list[float]` | EM log-likelihood per iteration |

### `MixturePrior`

| Field | Shape | Description |
|---|---|---|
| `weights` | `(k,)` | Mixture weights |
| `variances` | `(k,)` | Component variances |
| `mean_variance` | scalar | Weighted mean variance |

---

## Fitting APIs

### `fit_susie_univariate(X, y, L=10, prior_variance=1.0, residual_variance=None, estimate_residual_variance=True, max_iter=200, tol=1e-4) -> SusieResult`

- `X`: `(n, p)` — design matrix (will be mean-centered internally)
- `y`: `(n,)` — response vector

### `fit_susie_multivariate_independent(X, Y, ...) -> MultiSusieResult`

- `X`: `(n, p)`
- `Y`: `(n, r)` — fits each column independently

### `fit_susie_multivariate_joint(X, Y, L=10, prior_variance=1.0, residual_variance=None, estimate_residual_variance=True, estimate_prior_variance=False, max_iter=200, tol=1e-4) -> JointMultiSusieResult`

Full IBSS-M with shared `(r, r)` residual covariance Σ updated each iteration.

- `X`: `(n, p)`
- `Y`: `(n, r)`
- `estimate_residual_variance`: update Σ via `(Y - XB)'(Y - XB) / n` each iteration
- `estimate_prior_variance`: update scalar V via EM M-step each iteration

### `mvsusie(X, y, L=10, prior_variance=1.0, mixture_prior=None, residual_variance=None, estimate_residual_variance=True, max_iter=200, tol=1e-4, joint=False) -> SusieResult | MultiSusieResult | JointMultiSusieResult`

Unified dispatcher:

| `y` shape | `joint` | Returns |
|---|---|---|
| `(n,)` | — | `SusieResult` |
| `(n, r)` | `False` | `MultiSusieResult` (independent) |
| `(n, r)` | `True` | `JointMultiSusieResult` |

### `mvsusie_suff_stat(XtX, Xty, yty, n, L=10, prior_variance=1.0, residual_variance=None, estimate_residual_variance=True, max_iter=200, tol=1e-4) -> SusieResult | MultiSusieResult`

- `XtX`: `(p, p)`
- `Xty`: `(p,)` univariate or `(p, r)` multivariate
- `yty`: scalar (univariate) or `(r,)` (multivariate)
- `n`: sample size

### `mvsusie_rss(z, R, n, L=10, prior_variance=1.0, residual_variance=1.0, estimate_residual_variance=False, max_iter=200, tol=1e-4) -> SusieResult | MultiSusieResult`

- `z`: `(p,)` or `(p, r)` — marginal z-scores
- `R`: `(p, p)` — LD correlation matrix
- `n`: GWAS sample size

Internal conversion: `z_tilde = z / sqrt(1 + z²/n)`, `XtX = n·R`, `Xty = sqrt(n)·z_tilde`, `yty = n`.

### `mvsusie_rss_suff_stat(XtX, Xty, n, yty=None, ...) -> SusieResult | MultiSusieResult`

Convenience wrapper. If `yty=None`, defaults to `n` (univariate) or `[n]*r` (multivariate).

---

## Mash / Joint APIs

### `learn_mash_covariances(B_hat, S_hat=None, n_data_driven=3, max_iter=500, tol=1e-6, min_weight=1e-8) -> MashCovariance`

EM over mixture `Y ~ Σ_k π_k N(0, U_k)`. Components include:
- Null (zeros)
- Identity (shared effects across all traits)
- Rank-1 per condition `e_j e_j'` (trait-specific effects)
- Top `n_data_driven` PCA directions from `B_hat`

Parameters:
- `B_hat`: `(p, r)` — posterior effect estimates (e.g. from `coef(fit)`)
- `S_hat`: `(p, r)` — optional standard errors for scaling before PCA
- `n_data_driven`: number of data-driven PCA components

### `mash_reweight_joint(fit, mc, X, Y) -> JointMultiSusieResult`

Re-runs one IBSS-M pass replacing the scalar prior `V·I` with the learned mixture `Σ_k π_k · s · U_k` over a scale grid `[0.25, 0.5, 1.0, 2.0, 4.0]`. Log-BFs are computed per component and marginalised by log-sum-exp.

- `fit`: `JointMultiSusieResult`
- `mc`: `MashCovariance`
- `X`, `Y`: original data matrices

### `em_update_prior_variance(alpha, mu, mu2) -> float`

One EM M-step for scalar prior V:  `V = mean_l( Σ_j alpha_lj · E[b_lj²] )`.

Works for both univariate (`mu` shape `(L, p)`) and multivariate (`(L, p, r)`, averaged over r).

---

## Prior API

### `create_mixture_prior(weights, variances, normalize_weights=True) -> MixturePrior`

- `weights`: `(k,)`, non-negative
- `variances`: `(k,)`, positive
- `normalize_weights=False`: raises if weights don't sum to 1

---

## Utility APIs

### `coef(fit, include_intercept=False)`

| `fit` type | `include_intercept=False` | `include_intercept=True` |
|---|---|---|
| `SusieResult` | `(p,)` | `(p+1,)` |
| `MultiSusieResult` | `(p, r)` | `(p+1, r)` |
| `JointMultiSusieResult` | `(p, r)` via einsum | — |

### `predict(fit, X_new)`

- Univariate: `(n_new,)`
- Multivariate: `(n_new, r)`

### `calc_z(X, y) -> np.ndarray`

Compute marginal z-scores from raw data. Returns `(p,)`.

### `mvsusie_get_lfsr(fit) -> np.ndarray`

Minimum local false sign rate across effects. Returns `(p,)` or `(p, r)`.

### `mvsusie_single_effect_lfsr(fit) -> np.ndarray`

Per-effect LFSR. Returns `(L, p)` or `(L, p, r)`.

---

## CLI

Installed as `mvsusie` command after `pip install -e .`.

### `mvsusie dense`

```
mvsusie dense --X <path> --Y <path> [options]

Required:
  --X     Genotype matrix (.npy, .txt, .csv, .tsv)
  --Y     Phenotype matrix (.npy, .txt, .csv, .tsv)

Options:
  --L INT                      Number of single effects (default: 10)
  --prior-variance FLOAT       Initial prior variance V (default: 1.0)
  --joint                      Use joint multivariate inference
  --estimate-prior-variance    Learn V via EM (joint mode only)
  --no-estimate-residual-variance  Fix residual variance
  --mash-reweight              Learn mash covariances and reweight (joint only)
  --mash-n-data-driven INT     PCA components for mash (default: 3)
  --residual-variance PATH     Initial (r×r) Σ matrix file
  --max-iter INT               Max iterations (default: 200)
  --tol FLOAT                  Convergence tolerance (default: 1e-4)
  --out DIR                    Output directory (default: mvsusie_out/)
  --format {npy,txt}           Output format (default: txt)
```

Outputs: `pip`, `coef`, `alpha`, `meta.json`. With `--mash-reweight`: also `mash_weights`, `mash_components`.

### `mvsusie rss`

```
mvsusie rss --z <path> --R <path> --n INT [options]
```

### `mvsusie suff-stat`

```
mvsusie suff-stat --XtX <path> --Xty <path> --yty <path> --n INT [options]
```
