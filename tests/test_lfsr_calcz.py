import numpy as np

from mvsusie_py import (
    calc_z,
    fit_susie_univariate,
    mvsusie_get_lfsr,
    mvsusie_single_effect_lfsr,
)


def test_calc_z_ranks_strong_signal_highest():
    rng = np.random.default_rng(99)
    n, p = 250, 70
    causal = 11

    X = rng.normal(size=(n, p))
    b = np.zeros(p)
    b[causal] = 3.0
    y = X @ b + rng.normal(scale=1.0, size=n)

    z = calc_z(X, y)
    assert z.shape == (p,)
    assert int(np.argmax(np.abs(z))) == causal


def test_lfsr_shapes_and_ranges():
    rng = np.random.default_rng(1234)
    X = rng.normal(size=(160, 45))
    y = rng.normal(size=160)

    fit = fit_susie_univariate(X, y, L=5, max_iter=80)

    se_lfsr = mvsusie_single_effect_lfsr(fit)
    lfsr = mvsusie_get_lfsr(fit)

    assert se_lfsr.shape == (5, 45)
    assert lfsr.shape == (45,)
    assert np.all((se_lfsr >= 0) & (se_lfsr <= 1))
    assert np.all((lfsr >= 0) & (lfsr <= 1))
    assert np.allclose(lfsr, np.min(se_lfsr, axis=0))
