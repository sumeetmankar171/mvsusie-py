import numpy as np

from mvsusie_py import mvsusie, mvsusie_get_lfsr, mvsusie_single_effect_lfsr


def test_multivariate_lfsr_shapes_ranges_and_consistency():
    rng = np.random.default_rng(404)
    n, p, r = 200, 50, 3

    X = rng.normal(size=(n, p))
    B = np.zeros((p, r))
    B[[2, 7], 0] = [2.2, -1.3]
    B[[3, 10], 1] = [1.9, 1.1]
    B[[5, 12], 2] = [-2.1, 0.9]
    Y = X @ B + rng.normal(scale=1.0, size=(n, r))

    fit = mvsusie(
        X,
        Y,
        L=6,
        residual_variance=np.ones(r),
        estimate_residual_variance=False,
        max_iter=150,
        tol=1e-7,
    )

    se_lfsr = mvsusie_single_effect_lfsr(fit)
    lfsr = mvsusie_get_lfsr(fit)

    assert se_lfsr.shape == (6, p, r)
    assert lfsr.shape == (p, r)
    assert np.all((se_lfsr >= 0) & (se_lfsr <= 1))
    assert np.all((lfsr >= 0) & (lfsr <= 1))
    assert np.allclose(lfsr, np.min(se_lfsr, axis=0))
