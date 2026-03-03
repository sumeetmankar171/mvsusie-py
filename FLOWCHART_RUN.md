# Program Run Flowchart

Visual overview of all run paths in `mvsusie-py`.

## Flowchart

```mermaid
flowchart TD
    A[Start] --> B[Clone repo and install]
    B --> C{Choose fitting mode}

    C -->|Dense X/Y| D{Response type}
    C -->|Sufficient stats XtX/Xty/yty| G[mvsusie_suff_stat]
    C -->|RSS z/R| H[mvsusie_rss]
    C -->|RSS suff-stat| I[mvsusie_rss_suff_stat]

    D -->|Univariate y (n,)| E[mvsusie -> SusieResult]
    D -->|Multivariate Y (n,r) independent| F[mvsusie -> MultiSusieResult]
    D -->|Multivariate Y (n,r) joint=True| J[fit_susie_multivariate_joint -> JointMultiSusieResult]

    J --> K{Estimate prior variance?}
    K -->|estimate_prior_variance=True| L[EM update V each iteration]
    K -->|False| M[Fixed V]
    L --> N[learn_mash_covariances]
    M --> N

    N --> O[mash_reweight_joint -> refined JointMultiSusieResult]

    E --> P[coef / predict / pip]
    F --> P
    G --> P
    H --> P
    I --> P
    O --> P

    P --> Q[Validate: pytest + parity_smoke.py]
    Q --> R[Done]
```

## Step-by-step

### 1. Environment setup

```bash
git clone https://github.com/sumeetmankar171/mvsusie-py.git
cd mvsusie-py
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### 2. Choose fitting mode

| You have | Use |
|---|---|
| Raw `X` and `y`/`Y` | `mvsusie(X, Y, joint=True)` or `fit_susie_multivariate_joint` |
| Pre-computed `XtX`, `Xty`, `yty`, `n` | `mvsusie_suff_stat` |
| GWAS z-scores and LD matrix | `mvsusie_rss` |
| RSS-style sufficient stats | `mvsusie_rss_suff_stat` |

### 3. Joint multivariate mode (recommended for multi-trait plant eQTL)

```python
fit  = fit_susie_multivariate_joint(X, Y, L=10, estimate_prior_variance=True)
B_hat = np.einsum('lp,lpr->pr', fit.alpha, fit.mu)
mc   = learn_mash_covariances(B_hat, n_data_driven=3)
fit2 = mash_reweight_joint(fit, mc, X, Y)
```

Or with CLI:
```bash
mvsusie dense --X geno.npy --Y pheno.npy --joint \
              --estimate-prior-variance --mash-reweight --out results/
```

### 4. Post-fit outputs

```python
fit.pip                              # (p,) marginal PIPs
coef(fit)                            # (p,) or (p, r) posterior coefficients
coef(fit, include_intercept=True)    # with intercept
predict(fit, X_new)                  # (n_new,) or (n_new, r)
mvsusie_get_lfsr(fit)                # (p,) or (p, r) local false sign rates
```

### 5. Validation

```bash
PYTHONPATH=src python -m pytest -q            # 75 tests
PYTHONPATH=src python scripts/parity_smoke.py # dense/suff-stat/rss agreement
```

### 6. Common errors

| Error | Fix |
|---|---|
| `ModuleNotFoundError: mvsusie_py` | `pip install -e ".[dev]"` or set `PYTHONPATH=src` |
| Shape mismatch | Check `X` rows = `y`/`Y` rows; `XtX` square; `z` length = `R` size |
| Non-positive variance | Use positive `prior_variance` and `residual_variance` |
| Singular Σ warning | Normal — joint fitter applies `1e-6·I` ridge automatically |
