#!/usr/bin/env python3
"""
Quick parity smoke check for mvsusie-py interfaces.

Compares:
- mvsusie (dense)
- mvsusie_suff_stat
- mvsusie_rss

Outputs a compact JSON-like summary.
"""

from __future__ import annotations

import json
import numpy as np

from mvsusie_py import calc_z, coef, mvsusie, mvsusie_rss, mvsusie_suff_stat


def main() -> None:
    rng = np.random.default_rng(42)
    n, p = 400, 120
    causal = [9, 47]

    X = rng.normal(size=(n, p))
    b = np.zeros(p)
    b[causal] = [2.0, -1.5]
    y = X @ b + rng.normal(scale=1.0, size=n)

    fit_dense = mvsusie(
        X,
        y,
        L=8,
        prior_variance=1.0,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=200,
        tol=1e-7,
    )

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    fit_ss = mvsusie_suff_stat(
        XtX=Xc.T @ Xc,
        Xty=Xc.T @ yc,
        yty=float(yc @ yc),
        n=n,
        L=8,
        prior_variance=1.0,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=200,
        tol=1e-7,
    )

    z = calc_z(X, y)
    R = np.corrcoef(Xc, rowvar=False)
    fit_rss = mvsusie_rss(
        z=z,
        R=R,
        n=n,
        L=8,
        prior_variance=1.0,
        residual_variance=1.0,
        estimate_residual_variance=False,
        max_iter=200,
        tol=1e-7,
    )

    b_dense = coef(fit_dense)
    b_ss = coef(fit_ss)
    b_rss = coef(fit_rss)

    summary = {
        "top_pip": {
            "dense": int(np.argmax(fit_dense.pip)),
            "suff_stat": int(np.argmax(fit_ss.pip)),
            "rss": int(np.argmax(fit_rss.pip)),
        },
        "coef_corr": {
            "dense_vs_ss": float(np.corrcoef(b_dense, b_ss)[0, 1]),
            "dense_vs_rss": float(np.corrcoef(b_dense, b_rss)[0, 1]),
        },
        "causal": causal,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
