# mvsusie-py Run Guide

Step-by-step setup, all options explained, and common workflows.

## 1. What this package does

`mvsusie-py` is a pure Python/NumPy implementation of `mvsusieR` fine-mapping:

- Dense data, sufficient statistics, and RSS fine-mapping
- Univariate and multivariate outcomes
- **Joint multivariate** inference with shared residual covariance Σ
- **Mash-style** covariance mixture learning and posterior reweighting
- **EM-based** prior variance estimation
- **CLI** for file-based workflows

## 2. Prerequisites

- Python ≥ 3.9
- pip
- numpy ≥ 1.23
- pytest ≥ 7.0 (for tests)

## 3. Step-by-step setup

Clone and install from the project root:

```bash
git clone https://github.com/sumeetmankar171/mvsusie-py.git
cd mvsusie-py
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Explanation:
- `python3 -m venv .venv` — creates an isolated Python environment
- `source .venv/bin/activate` — activates it
- `pip install -e ".[dev]"` — editable install + dev deps (pytest)

## 4. Validate the install

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/parity_smoke.py
```

Expected: `75 passed` and a JSON parity summary showing top-PIP agreement across dense/suff-stat/rss modes.

## 5. Recommended workflows

### 5.1 Joint multi-trait eQTL (full pipeline)

```python
import numpy as np
from mvsusie_py import fit_susie_multivariate_joint, learn_mash_covariances, mash_reweight_joint

X = np.load("geno.npy")   # (n, p) genotype matrix
Y = np.load("pheno.npy")  # (n, r) expression matrix

# Step 1: joint fit with EM prior variance learning
fit = fit_susie_multivariate_joint(
    X, Y,
    L=10,
    estimate_prior_variance=True,
    estimate_residual_variance=True,
    max_iter=200,
    tol=1e-4,
)
print(fit.pip)                 # (p,) shared PIPs
print(fit.residual_variance)   # (r, r) learned Σ
print(fit.prior_variance)      # learned V

# Step 2: learn mash covariance mixture
B_hat = np.einsum('lp,lpr->pr', fit.alpha, fit.mu)
mc = learn_mash_covariances(B_hat, n_data_driven=3)
print(mc.weights)              # (K,) dominant patterns

# Step 3: reweight with learned structure
fit2 = mash_reweight_joint(fit, mc, X, Y)
print(fit2.pip)                # refined cross-trait PIPs
```

### 5.2 Joint fit via CLI

```bash
# Full pipeline in one command
mvsusie dense \
  --X geno.npy \
  --Y pheno.npy \
  --L 10 \
  --joint \
  --estimate-prior-variance \
  --mash-reweight \
  --mash-n-data-driven 3 \
  --format npy \
  --out results/

# Outputs in results/:
#   pip.npy          (p,) marginal PIPs
#   coef.npy         (p, r) posterior coefficients
#   alpha.npy        (L, p) single-effect probabilities
#   mash_weights.npy (K,) mixture weights
#   mash_components.npy (K, r, r) covariance matrices
#   meta.json        convergence info
```

### 5.3 Univariate dense fit

```python
from mvsusie_py import mvsusie, coef, predict

fit = mvsusie(X, y, L=8, prior_variance=1.0, estimate_residual_variance=True)
print(coef(fit).shape)                        # (p,)
print(coef(fit, include_intercept=True).shape) # (p+1,)
print(predict(fit, X).shape)                   # (n,)
```

### 5.4 Sufficient-stat fit (large datasets)

```python
from mvsusie_py import mvsusie_suff_stat

Xc  = X - X.mean(axis=0)
yc  = y - y.mean()
fit = mvsusie_suff_stat(
    XtX=Xc.T @ Xc,
    Xty=Xc.T @ yc,
    yty=float(yc @ yc),
    n=n,
    L=10,
)
```

Or via CLI:
```bash
mvsusie suff-stat --XtX XtX.npy --Xty Xty.npy --yty yty.npy --n 500 --out results/
```

### 5.5 RSS fine-mapping from GWAS summary stats

```python
from mvsusie_py import mvsusie_rss, calc_z

z   = calc_z(X, y)
R   = np.corrcoef((X - X.mean(axis=0)), rowvar=False)
fit = mvsusie_rss(z=z, R=R, n=n, L=10)
```

Or via CLI:
```bash
mvsusie rss --z zscores.npy --R ld.npy --n 500 --out results/
```

## 6. All options explained

### `fit_susie_multivariate_joint`

| Option | Default | Description |
|---|---|---|
| `L` | 10 | Number of single-effect components |
| `prior_variance` | 1.0 | Initial scalar prior variance V |
| `residual_variance` | None | Initial (r×r) Σ; defaults to `diag(var(Y))` |
| `estimate_residual_variance` | True | Update Σ each IBSS-M iteration |
| `estimate_prior_variance` | False | Update V via EM M-step each iteration |
| `max_iter` | 200 | Maximum IBSS-M iterations |
| `tol` | 1e-4 | Convergence: stop when `‖B_new - B_old‖ < tol` |

### `learn_mash_covariances`

| Option | Default | Description |
|---|---|---|
| `n_data_driven` | 3 | Number of PCA components from B_hat |
| `max_iter` | 500 | EM iterations |
| `tol` | 1e-6 | EM convergence tolerance |
| `min_weight` | 1e-8 | Floor for mixture weights |

### `mvsusie` (unified dispatcher)

| Option | Default | Description |
|---|---|---|
| `L` | 10 | Single effects |
| `prior_variance` | 1.0 | Scalar prior variance |
| `mixture_prior` | None | `MixturePrior`; uses `mean_variance` as effective V |
| `residual_variance` | None | Scalar, `(r,)`, or `(r,r)`; if None, auto-initialized |
| `estimate_residual_variance` | True | Update each iteration |
| `joint` | False | Use joint inference for multivariate Y |
| `max_iter` | 200 | Max iterations |
| `tol` | 1e-4 | Convergence tolerance |

### CLI shared options

| Flag | Default | Description |
|---|---|---|
| `--L` | 10 | Single effects |
| `--prior-variance` | 1.0 | Initial V |
| `--max-iter` | 200 | Max iterations |
| `--tol` | 1e-4 | Convergence tolerance |
| `--no-estimate-residual-variance` | off | Fix residual variance |
| `--out` | `mvsusie_out/` | Output directory |
| `--format` | `txt` | `npy` (binary) or `txt` |

## 7. Troubleshooting

**`ModuleNotFoundError: mvsusie_py`**
```bash
pip install -e ".[dev]"
# or run with explicit path:
PYTHONPATH=src python -m pytest -q
```

**Shape errors**
- `X` rows must match `y`/`Y` rows
- `XtX` must be square with diagonal > 0
- `Xty` first dimension must equal `XtX.shape[0]`
- `z` first dimension must equal `R.shape[0]`
- `residual_variance` passed to joint fit must be `(r, r)`

**Singular residual covariance**
- The joint fitter applies a `1e-6 · I` ridge by default — safe to ignore warning
- For very small `r` or degenerate data, try increasing `tol` or reducing `L`

**EM prior variance instability**
- If `prior_variance_trace` oscillates, reduce `max_iter` or tighten `tol`
- Default `estimate_prior_variance=False` (stable scalar V) is safe for most use cases
