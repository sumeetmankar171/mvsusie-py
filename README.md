# mvsusie-py

Python implementation of core `mvsusieR` fine-mapping workflows — no R or GSL dependency required.

## What you get

**Fitting modes**
- Dense-data fitting: `mvsusie(X, Y, joint=True)`
- Sufficient-stat fitting: `mvsusie_suff_stat(XtX, Xty, yty, n)`
- RSS fine-mapping: `mvsusie_rss(z, R, n)`
- RSS sufficient-stat wrapper: `mvsusie_rss_suff_stat(...)`

**Joint multivariate inference**
- `fit_susie_multivariate_joint` — full IBSS-M with shared (r×r) residual covariance Σ
- `em_update_prior_variance` — EM-based learning of scalar prior variance V
- `learn_mash_covariances` — data-driven mash-style covariance mixture (EM over canonical + PCA components)
- `mash_reweight_joint` — feed learned covariance structure back into IBSS as a structured prior

**Utilities**
- `coef`, `predict`, `calc_z`
- `mvsusie_get_lfsr`, `mvsusie_single_effect_lfsr`
- `create_mixture_prior`

**CLI**
```bash
mvsusie dense     --X geno.npy --Y pheno.npy --joint --estimate-prior-variance --mash-reweight --out results/
mvsusie rss       --z zscores.npy --R ld.npy --n 500 --out results/
mvsusie suff-stat --XtX XtX.npy --Xty Xty.npy --yty yty.npy --n 500 --out results/
```

## Project status

- Version: `0.3.0`
- Tests: **75 passed**
- CI: GitHub Actions (Python 3.9–3.12)

## Installation

```bash
git clone https://github.com/sumeetmankar171/mvsusie-py.git
cd mvsusie-py
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

### Univariate fine-mapping

```python
import numpy as np
from mvsusie_py import mvsusie, coef, predict

rng = np.random.default_rng(1)
X = rng.normal(size=(200, 100))
b = np.zeros(100); b[5] = 3.0
y = X @ b + rng.normal(size=200)

fit = mvsusie(X, y, L=8)
print(fit.pip.shape)          # (100,) — posterior inclusion probabilities
print(coef(fit).shape)        # (100,)
```

### Joint multivariate fine-mapping (recommended for multi-trait eQTL)

```python
import numpy as np
from mvsusie_py import fit_susie_multivariate_joint, learn_mash_covariances, mash_reweight_joint

X = np.load("geno.npy")   # (n, p)
Y = np.load("pheno.npy")  # (n, r)

# Step 1: joint fit with EM prior variance
fit = fit_susie_multivariate_joint(X, Y, L=10, estimate_prior_variance=True)
print(fit.pip)                # (p,) shared PIPs across traits
print(fit.residual_variance)  # (r, r) learned Σ

# Step 2: learn mash covariances from posterior effects
B_hat = np.einsum('lp,lpr->pr', fit.alpha, fit.mu)
mc    = learn_mash_covariances(B_hat, n_data_driven=3)
print(mc.weights)             # (K,) which cross-trait patterns dominate

# Step 3: reweight posteriors using learned structure
fit2  = mash_reweight_joint(fit, mc, X, Y)
print(fit2.pip)               # refined PIPs informed by cross-trait covariance
```

### RSS fine-mapping from GWAS summary stats

```python
from mvsusie_py import mvsusie_rss

fit = mvsusie_rss(z=z, R=R, n=500, L=10)
print(fit.pip)
```

## Validate install

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/parity_smoke.py
```

## Documentation

| Document | Contents |
|---|---|
| [API.md](./API.md) | Full API signatures, shapes, return types |
| [RUN_GUIDE.md](./RUN_GUIDE.md) | Step-by-step setup and all options explained |
| [FLOWCHART_RUN.md](./FLOWCHART_RUN.md) | Visual run flowchart |
| [CHANGELOG.md](./CHANGELOG.md) | Version history |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | Development and PR process |
| [RELEASE_CHECKLIST.md](./RELEASE_CHECKLIST.md) | Release steps |

## Design

- Pure Python/NumPy — no R, no GSL, no C extensions
- IBSS-M algorithm: iterative Bayesian stepwise selection with shared multivariate residual covariance
- Mash-style prior: EM over canonical (identity, rank-1 per trait) + data-driven (PCA) covariance components
- Scalar prior variance V updated via closed-form EM M-step
- All three fitting modes (dense, suff-stat, RSS) share the same IBSS core

## License

MIT. See [LICENSE](./LICENSE).
