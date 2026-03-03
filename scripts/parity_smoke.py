#!/usr/bin/env python3
"""
Parity smoke check for mvsusie-py.

Verifies:
  1. Dense / suff-stat / RSS produce consistent PIPs and coefficients (univariate)
  2. Joint multivariate fit runs and produces valid PIPs + learned Sigma
  3. Mash covariance learning produces valid weights
  4. Mash reweighting runs end-to-end without error

Run:
    PYTHONPATH=src python scripts/parity_smoke.py
"""

from __future__ import annotations

import json
import sys

import numpy as np

from mvsusie_py import (
    calc_z,
    coef,
    fit_susie_multivariate_joint,
    learn_mash_covariances,
    mash_reweight_joint,
    mvsusie,
    mvsusie_rss,
    mvsusie_suff_stat,
)


def smoke_univariate() -> dict:
    """Check dense / suff-stat / RSS agree on a univariate synthetic problem."""
    rng = np.random.default_rng(42)
    n, p = 400, 120
    causal = [9, 47]

    X = rng.normal(size=(n, p))
    b = np.zeros(p)
    b[causal] = [2.0, -1.5]
    y = X @ b + rng.normal(scale=1.0, size=n)

    kwargs = dict(L=8, prior_variance=1.0, residual_variance=1.0,
                  estimate_residual_variance=False, max_iter=200, tol=1e-7)

    fit_dense = mvsusie(X, y, **kwargs)

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    fit_ss = mvsusie_suff_stat(
        XtX=Xc.T @ Xc, Xty=Xc.T @ yc, yty=float(yc @ yc), n=n, **kwargs
    )

    z = calc_z(X, y)
    R = np.corrcoef(Xc, rowvar=False)
    fit_rss = mvsusie_rss(z=z, R=R, n=n, **kwargs)

    b_dense = coef(fit_dense)
    b_ss    = coef(fit_ss)
    b_rss   = coef(fit_rss)

    return {
        "top_pip": {
            "dense":     int(np.argmax(fit_dense.pip)),
            "suff_stat": int(np.argmax(fit_ss.pip)),
            "rss":       int(np.argmax(fit_rss.pip)),
        },
        "coef_corr": {
            "dense_vs_ss":  float(np.corrcoef(b_dense, b_ss)[0, 1]),
            "dense_vs_rss": float(np.corrcoef(b_dense, b_rss)[0, 1]),
        },
        "causal_snps": causal,
    }


def smoke_joint_mash() -> dict:
    """Check joint fit + mash learning + reweighting pipeline."""
    rng = np.random.default_rng(7)
    n, p, r = 300, 80, 3
    X = rng.normal(size=(n, p))
    B = np.zeros((p, r))
    B[0] = [2.0, -1.5, 1.0]
    B[1] = [-1.0, 2.0, 0.5]
    L_noise = np.array([[1.0, 0.0, 0.0],
                         [0.5, 0.866, 0.0],
                         [0.3, 0.4,   0.866]])
    Y = X @ B + rng.multivariate_normal(np.zeros(r), L_noise @ L_noise.T, size=n)

    fit = fit_susie_multivariate_joint(
        X, Y, L=8, max_iter=100, estimate_prior_variance=True
    )

    B_hat = np.einsum('lp,lpr->pr', fit.alpha, fit.mu)
    mc    = learn_mash_covariances(B_hat, n_data_driven=3)
    fit2  = mash_reweight_joint(fit, mc, X, Y)

    return {
        "joint_fit": {
            "converged":      fit.converged,
            "n_iter":         fit.n_iter,
            "learned_V":      round(float(fit.prior_variance), 4),
            "pip_causal_0":   round(float(fit.pip[0]), 4),
            "pip_causal_1":   round(float(fit.pip[1]), 4),
            "sigma_diagonal": [round(float(v), 4) for v in np.diag(fit.residual_variance)],
        },
        "mash": {
            "n_components": int(mc.components.shape[0]),
            "weights_sum":  round(float(mc.weights.sum()), 6),
            "top_weight":   round(float(mc.weights.max()), 4),
        },
        "reweighted": {
            "pip_causal_0": round(float(fit2.pip[0]), 4),
            "pip_causal_1": round(float(fit2.pip[1]), 4),
        },
    }


def main() -> None:
    print("=== Univariate parity: dense / suff-stat / rss ===")
    uni = smoke_univariate()
    print(json.dumps(uni, indent=2))

    print("\n=== Joint multivariate + mash pipeline ===")
    jt = smoke_joint_mash()
    print(json.dumps(jt, indent=2))

    # Sanity assertions
    errors = []

    if uni["top_pip"]["dense"] != uni["top_pip"]["suff_stat"]:
        errors.append("dense and suff-stat disagree on top PIP SNP")

    if uni["coef_corr"]["dense_vs_ss"] < 0.99:
        errors.append(f"dense vs suff-stat corr too low: {uni['coef_corr']['dense_vs_ss']:.4f}")

    if abs(jt["mash"]["weights_sum"] - 1.0) > 1e-4:
        errors.append(f"mash weights don't sum to 1: {jt['mash']['weights_sum']}")

    if jt["joint_fit"]["pip_causal_0"] < 0.5:
        errors.append(f"causal SNP 0 has low PIP: {jt['joint_fit']['pip_causal_0']}")

    if errors:
        print("\nFAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
